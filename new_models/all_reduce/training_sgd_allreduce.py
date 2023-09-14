#! /usr/bin/env python3


import argparse
import copy
import logging
import os
import queue
import random
import threading
import time

import numpy as np
import pandas as pd
import psutil
import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
from models import (
    alexnet,
    densenet,
    googlenet,
    lstm,
    mobilenetv2,
    resnet3,
    transformer,
    vgg,
    vit,
)
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator
from torchvision import datasets, transforms
from transformers import BertModel, BertTokenizer, GPT2Model, GPT2Tokenizer
from zeyu_utils import net as znet

rand_seed = 218276150
torch.manual_seed(rand_seed)
torch.cuda.manual_seed(rand_seed)
torch.cuda.manual_seed_all(rand_seed)
np.random.seed(rand_seed)
random.seed(rand_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def data_process(raw_text_iter, vocab, tokenizer):
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].reshape(-1)
    return data, target


class BBCTextDataset(torch.utils.data.Dataset):
    def __init__(self, df, model_name="gpt"):
        if model_name == "gpt":
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
        elif model_name == "bert":
            tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        labels = {"business": 0, "entertainment": 1, "sport": 2, "tech": 3, "politics": 4}
        self.labels = [labels[label] for label in df["category"]]
        self.texts = [
            tokenizer(text, padding="max_length", max_length=384, truncation=True, return_tensors="pt") for text in df["text"]
        ]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


class BertClassifier(torch.nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(768, 5)
        self.relu = torch.nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


class GPT2Classifier(torch.nn.Module):
    def __init__(self, hidden_size=768, max_seq_len=384):
        super(GPT2Classifier, self).__init__()

        self.gpt2 = GPT2Model.from_pretrained("gpt2")
        # self.dropout = torch.nn.Dropout(0.5)
        self.linear = torch.nn.Linear(max_seq_len * hidden_size, 5)
        self.relu = torch.nn.ReLU()

    def forward(self, input_id, mask):
        gpt2_out, _ = self.gpt2(input_ids=input_id, attention_mask=mask, return_dict=False)
        if len(gpt2_out.shape) > 2:
            batch_size = gpt2_out.shape[0]
        else:
            batch_size = 1
        # linear_input = self.dropout(gpt2_out.view(batch_size, -1))
        linear_output = self.linear(gpt2_out.view(batch_size, -1))
        final_layer = self.relu(linear_output)

        return final_layer


class Logger(object):
    def __init__(self, job_name, file_path, log_level=logging.INFO, mode="w"):
        self.__logger = logging.getLogger(job_name)
        self.__logger.setLevel(log_level)
        self.__fh = logging.FileHandler(filename=file_path, mode=mode)
        self.__formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
        self.__fh.setFormatter(self.__formatter)
        self.__logger.addHandler(self.__fh)

    @property
    def logger(self):
        return self.__logger


class Worker:
    def __init__(
        self,
        wrk_id,
        job_name,
        model_name,
        model,
        wrk_num,
        epoch_num,
        learning_rate,
        gpu_id,
        training_data_dir,
        batch_size,
        data_loader,
        wrk_pkey_lists,
        wrk_pkey_locs,
    ) -> None:
        self.wrk_id = wrk_id
        self.ring_wrk_id = wrk_id
        self.job_name = job_name
        self.model_name = model_name
        self.model = model
        self.model_lock = threading.Lock()
        self.wrk_num = wrk_num
        self.ring_wrk_num = wrk_num
        self.epoch_num = epoch_num
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.training_data_dir = training_data_dir
        self.batch_size = batch_size
        self.data_loader = data_loader
        self.wrk_pkey_lists = wrk_pkey_lists
        self.wrk_pkey_locs = wrk_pkey_locs
        self.grad_lists = None
        self.wrk_rrefs = None
        self.pre_wrk_rref = None
        self.tester_rref = None
        self.logger = Logger(job_name=job_name, file_path=f"./training_logs/{job_name}_worker{wrk_id}.log").logger
        self.logger.info(f"Model name: {model_name}")

        self.epoch_idx = 0
        self.iter_idx = 0
        self.iter_count = 0

        # aggregate
        self.aggr_msg_q = queue.Queue()

        # param update
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        self.future_iter_sync = None
        self.is_ring_wrk = True

        self.step_num = 0

    def update_optimizer(self, lr):
        if not isinstance(self, Worker):
            self = self.local_value()
        with self.model_lock:
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

    def set_wrk_rrefs_and_tester_rref(self, worker_rrefs, tester_rref):
        if not isinstance(self, Worker):
            self = self.local_value()

        self.wrk_rrefs = worker_rrefs
        self.pre_wrk_rref = worker_rrefs[(self.wrk_id - 1) % self.wrk_num]
        self.tester_rref = tester_rref

    def get_model(self):
        if not isinstance(self, Worker):
            self = self.local_value()

        if self.model_name == "transformer":
            params = []
            with self.model_lock:
                for key, _ in self.model.named_parameters():
                    params.append(self.model.state_dict()[key].cpu())
            return params

        model = None
        with self.model_lock:
            self.model = self.model.cpu()
            model = copy.deepcopy(self.model)
            self.model = self.model.to(self.device)

        return model, self.step_num

    def get_gradient_slice(self):
        if not isinstance(self, Worker):
            self = self.local_value()

        grad_slice = self.aggr_msg_q.get()

        return grad_slice, time.time()

    def run_worker(self):
        if not isinstance(self, Worker):
            self = self.local_value()

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(self.device)

        self.model = self.model.to(self.device)

        torch.cuda.set_device(int(self.gpu_id))

        train_iter = WikiText2(root=self.training_data_dir, split="train")
        tokenizer = get_tokenizer("basic_english")
        vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        ntokens = len(vocab)
        bptt = 35

        train_iter, val_iter, test_iter = WikiText2(root=self.training_data_dir)
        train_data = data_process(train_iter, vocab, tokenizer)
        train_data = batchify(train_data, self.batch_size)

        psutil.cpu_percent()
        bw0 = psutil.net_io_counters()

        self.model.train()
        for epoch_idx in range(self.epoch_num):
            self.epoch_idx = epoch_idx

            if self.model_name == "lstm":
                hidden = self.model.init_hidden(self.batch_size)

            if self.model_name == "lstm" or self.model_name == "transformer":
                enumerator = enumerate(range(0, train_data.size(0) - 1, bptt))
            else:
                enumerator = enumerate(self.data_loader)

            for iter_idx, x_pack in enumerator:
                self.iter_idx = iter_idx
                self.iter_count += 1

                self.step_num += 1

                if self.model_name == "lstm" or self.model_name == "transformer":
                    i = x_pack
                    data, target = get_batch(train_data, i, bptt)
                else:
                    data, target = x_pack

                wrk_cp_t_0 = time.time()

                if self.model_name == "bert" or self.model_name == "gpt":
                    target = target.to(self.device)
                    mask = data["attention_mask"].squeeze().to(self.device)
                    input_ids = data["input_ids"].squeeze().to(self.device)
                    self.model.zero_grad()
                    output = self.model(input_ids, mask)
                    batch_loss = criterion(output, target.long())
                    batch_loss.backward()
                else:
                    # other models
                    # self.model = self.model.to(device)
                    data, target = data.to(self.device), target.to(self.device)
                    self.model.zero_grad()
                    if self.model_name == "lstm":
                        hidden = repackage_hidden(hidden)
                        output, hidden = self.model(data, hidden)
                    elif self.model_name == "transformer":
                        output = self.model(data)
                        output = output.view(-1, ntokens)
                    else:
                        output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()

                wrk_cp_t = 1000 * (time.time() - wrk_cp_t_0)

                # prepare grad lists
                self.grad_lists = [[None for _ in range(len(self.wrk_pkey_lists[i]))] for i in range(self.ring_wrk_num)]
                with self.model_lock:
                    for pkey, param in self.model.named_parameters():
                        pkey_loc = self.wrk_pkey_locs[pkey]
                        self.grad_lists[pkey_loc[0]][pkey_loc[1]] = param.grad.cpu()
                # define wrk cm t
                self.wrk_cm_t = 0.0
                # exchange gradients with other workers
                for idx in range(self.ring_wrk_num - 1):
                    send_slice_id = (self.ring_wrk_id - idx) % self.ring_wrk_num
                    recv_slice_id = (send_slice_id - 1) % self.ring_wrk_num
                    send_grad_slice = self.grad_lists[send_slice_id]
                    self.aggr_msg_q.put(send_grad_slice)
                    recv_grad_slice, wrk_cm_t_0 = self.pre_wrk_rref.rpc_sync().get_gradient_slice()
                    self.wrk_cm_t += 1000 * (time.time() - wrk_cm_t_0)
                    grad_list_to_add = self.grad_lists[recv_slice_id]
                    for i in range(len(grad_list_to_add)):
                        grad_list_to_add[i] += recv_grad_slice[i]
                for idx in range(self.ring_wrk_num - 1):
                    send_slice_id = (self.ring_wrk_id + 1 - idx) % self.ring_wrk_num
                    recv_slice_id = (send_slice_id - 1) % self.ring_wrk_num
                    send_grad_slice = self.grad_lists[send_slice_id]
                    self.aggr_msg_q.put(send_grad_slice)
                    recv_grad_slice, wrk_cm_t_0 = self.pre_wrk_rref.rpc_sync().get_gradient_slice()
                    self.wrk_cm_t += 1000 * (time.time() - wrk_cm_t_0)
                    self.grad_lists[recv_slice_id] = recv_grad_slice
                # update model parameter
                with self.model_lock:
                    for pkey, param in self.model.named_parameters():
                        pkey_loc = self.wrk_pkey_locs[pkey]
                        grad = self.grad_lists[pkey_loc[0]][pkey_loc[1]].to(self.device)
                        param.grad = grad
                    self.optimizer.step()

                stop_flag = self.tester_rref.rpc_sync().get_stop_flag()

                if stop_flag:
                    break

            if stop_flag:
                break


class Tester:
    def __init__(
        self,
        job_name,
        model_name,
        model,
        worker_num,
        gpu_id,
        testing_data_dir,
        worker_rrefs,
        # model_fetch_ids,
        test_batch_size,
        test_target_loss,
        test_data_loader,
        test_dataset,
        test_wrk_id,
    ) -> None:
        self.job_name = job_name
        self.model_name = model_name
        self.model = model
        self.worker_num = worker_num
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        if model_name == "transformer":
            self.model = self.model.to(self.device)
        self.testing_data_dir = testing_data_dir
        self.worker_rrefs = worker_rrefs
        # self.model_fetch_ids = model_fetch_ids
        # self.cur_fetch_index = 0
        self.test_batch_size = test_batch_size
        self.test_target_loss = test_target_loss
        self.test_data_loader = test_data_loader
        self.test_dataset = test_dataset
        self.test_wrk_id = test_wrk_id
        self.logger = Logger(job_name=job_name, file_path=f"./training_logs/{job_name}_tester.log").logger
        self.logger.info(f"Model name: {model_name}")

        self.stop_flag = False

    def get_stop_flag(self):
        if not isinstance(self, Tester):
            self = self.local_value()

        return self.stop_flag

    def test_model(self, test_df=None):
        if not isinstance(self, Tester):
            self = self.local_value()

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(self.device)

        train_iter = WikiText2(root=self.testing_data_dir, split="train")
        tokenizer = get_tokenizer("basic_english")
        vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        ntokens = len(vocab)
        bptt = 35

        train_iter, val_iter, test_iter = WikiText2(root=self.testing_data_dir)
        val_data = data_process(val_iter, vocab, tokenizer)
        val_data = batchify(val_data, self.test_batch_size)

        time_init = time.time()
        time_0 = time_init
        self.model = self.model.to(self.device)
        self.model.eval()
        while True:
            time_1 = time.time()

            if time_1 - time_0 >= 40:
                time_0 = time_1

                if self.model_name == "transformer":
                    params = self.worker_rrefs[self.test_wrk_id].rpc_sync().get_model()
                    count = 0
                    for key, _ in self.model.named_parameters():
                        self.model.state_dict()[key].copy_(params[count])
                        count += 1
                else:
                    self.model, step_num = self.worker_rrefs[self.test_wrk_id].rpc_sync().get_model()
                    self.model = self.model.to(self.device)

                test_correct = 0.0
                test_loss = 0.0

                if self.model_name in ["bert", "gpt"]:
                    total_acc_val = 0
                    total_loss_val = 0
                    with torch.no_grad():
                        for test_input, test_label in self.test_data_loader:
                            test_label = test_label.to(self.device)
                            mask = test_input["attention_mask"].squeeze().to(self.device)
                            input_id = test_input["input_ids"].squeeze().to(self.device)
                            output = self.model(input_id, mask)
                            batch_loss = criterion(output, test_label.long())
                            total_loss_val += batch_loss.item() * len(test_label)
                            acc = (output.argmax(dim=1) == test_label).sum().item()
                            total_acc_val += acc
                    test_accuracy = total_acc_val / len(test_df) * 100
                    test_loss = total_loss_val / len(test_df)
                else:
                    if self.model_name == "lstm":
                        with torch.no_grad():
                            hidden = self.model.init_hidden(self.test_batch_size)
                            for _, i in enumerate(range(0, val_data.size(0) - 1, bptt)):
                                data, targets = get_batch(val_data, i, bptt)
                                data, targets = data.to(self.device), targets.to(self.device)
                                hidden = repackage_hidden(hidden)
                                output, hidden = self.model(data, hidden)
                                loss = criterion(output, targets)
                                test_loss += len(data) * loss.item()
                    elif self.model_name == "transformer":
                        with torch.no_grad():
                            for _, i in enumerate(range(0, val_data.size(0) - 1, bptt)):
                                data, targets = get_batch(val_data, i, bptt)
                                data, targets = data.to(self.device), targets.to(self.device)
                                output = self.model(data)
                                output = output.view(-1, ntokens)
                                loss = criterion(output, targets)
                                test_loss += len(data) * loss.item()
                    else:
                        with torch.no_grad():
                            for _, (data, target) in enumerate(self.test_data_loader):
                                data, target = data.to(self.device), target.to(self.device)
                                output = self.model(data)
                                loss = criterion(output, target)
                                test_loss += loss.item()
                                _, predicted = output.max(1)
                                test_correct += predicted.eq(target).sum().item()

                if self.model_name != "transformer":
                    self.model = self.model.cpu()

                if self.model_name not in ["bert", "gpt"]:
                    if self.model_name == "lstm" or self.model_name == "transformer":
                        test_loss /= len(val_data) - 1
                    else:
                        test_loss = test_loss * self.test_batch_size / len(self.test_data_loader.dataset)
                        test_accuracy = 100.0 * test_correct / len(self.test_dataset)

                step_num = self.ps_rrefs[0].rpc_sync().get_step_num()

                if self.model_name == "lstm" or self.model_name == "transformer":
                    self.logger.info(
                        "Steps: {} | Loss: {:.4f} | Time: {:.4f} s".format(step_num, test_loss, time_1 - time_init)
                    )
                else:
                    self.logger.info(
                        "Steps: {} | Loss: {:.4f} | Acc.: {:.4f} % | Time: {:.4f} s".format(
                            step_num, test_loss, test_accuracy, time_1 - time_init
                        )
                    )

                # if test_loss <= self.test_target_loss:
                #     self.ps_rref.rpc_sync().stop_ps()
                #     break


def main(
    job_name,
    model_name,
    rpc_rank,
    wrk_num,
    training_data_dir,
    batch_size,
    test_batch_size,
    test_target_loss,
    test_wrk_id,
    learning_rate,
    epoch_num,
    data_partitioned,
    gpu_ids,
    # model_fetch_ids,
):
    logging.basicConfig(level=logging.INFO)
    world_size = wrk_num + 2
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=16, rpc_timeout=0, _transports=["uv"])

    if rpc_rank == 0:  # manager
        # Partition parameters to PSs
        model = None
        if model_name == "alexnet":
            model = alexnet.AlexNet()
        elif model_name == "resnet20":
            model = resnet3.resnet20()
        elif model_name == "resnet56":
            model = resnet3.resnet56()
        elif model_name == "vgg13":
            model = vgg.VGG13()
        elif model_name == "vgg16":
            model = vgg.VGG16()
        elif model_name == "densenet121":
            model = densenet.DenseNet121()
        elif model_name == "googlenet":
            model = googlenet.GoogLeNet()
        elif model_name == "mobilenet":
            model = mobilenetv2.MobileNetV2()
        elif model_name == "vit":
            model = vit.ViT(
                image_size=32,
                patch_size=4,
                num_classes=10,
                dim=int(512),
                depth=6,
                heads=8,
                mlp_dim=512,
                dropout=0.1,
                emb_dropout=0.1,
            )
        elif model_name == "lstm":
            model = lstm.RNNModel(rnn_type="LSTM", ntoken=28782, ninp=200, nhid=200, nlayers=2, dropout=0)
        elif model_name == "transformer":
            model = transformer.TransformerModel(ntoken=28782, ninp=200, nhead=8, nhid=200, nlayers=2, dropout=0)
        elif model_name == "bert":
            model = BertClassifier()
        elif model_name == "gpt":
            model = GPT2Classifier()
        model.train()
        pkey_numel = {}
        key_param = {}
        wrk_param_nums = [0 for _ in range(wrk_num)]
        # ps_param_lists = [[] for _ in range(ps_num)]  # needed for ps
        # param_ps_idx = {}  # needed for ps and worker
        # param_loc_idx = {}  # needed for ps and worker
        wrk_pkey_lists = [[] for _ in range(wrk_num)]
        wrk_pkey_locs = {}
        for key, param in model.named_parameters():
            pkey_numel[key] = param.numel()
            key_param[key] = param
        pkey_numel = sorted(pkey_numel.items(), key=lambda x: x[1], reverse=True)
        for i in range(len(pkey_numel)):
            key = pkey_numel[i][0]
            idx = np.argmin(wrk_param_nums)
            # param_ps_idx[key] = idx
            # param_loc_idx[key] = len(ps_param_lists[idx])
            wrk_param_nums[idx] += pkey_numel[i][1]
            wrk_pkey_locs[key] = (idx, len(wrk_pkey_lists[idx]))
            wrk_pkey_lists[idx].append(key)
        # Parameter partitioning done

        # Initializing all data_loaders for all workers.
        test_df = None
        if model_name not in ["bert", "gpt"]:
            data_loaders = []
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
            training_dataset = datasets.CIFAR10(root=training_data_dir, train=True, download=True, transform=transform)
            if data_partitioned == 1:
                dataset_len = len(training_dataset)
                worker_dataset_len = int((dataset_len + wrk_num) / wrk_num)
                len_list = [worker_dataset_len for _ in range(wrk_num - 1)]
                len_list.append(dataset_len - (wrk_num - 1) * worker_dataset_len)
                training_datasets = random_split(training_dataset, len_list)
                for id in range(wrk_num):
                    data_loader = DataLoader(training_datasets[id], batch_size=batch_size, shuffle=True)
                    data_loaders.append(data_loader)
            else:
                for _ in range(wrk_num):
                    data_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
                    data_loaders.append(data_loader)
            # Initialized all data_loaders for all workers.
        else:
            # for bert and gpt loader
            df = pd.read_csv("./bbc-text.csv")
            train_df, test_df = np.split(
                df.sample(frac=1, random_state=42),
                [
                    int(0.8 * len(df)),
                ],
            )
            train_ds, test_ds = BBCTextDataset(train_df, model_name), BBCTextDataset(test_df, model_name)
            data_loaders = []
            train_ds_len = len(train_ds)
            wrk_ds_len = int((train_ds_len + wrk_num) / wrk_num)
            len_list = [wrk_ds_len for _ in range(wrk_num - 1)]
            len_list.append(train_ds_len - (wrk_num - 1) * wrk_ds_len)
            train_ds_list = torch.utils.data.random_split(train_ds, len_list)
            for idx in range(wrk_num):
                data_loader = torch.utils.data.DataLoader(train_ds_list[idx], batch_size=2, shuffle=True, drop_last=True)
                data_loaders.append(data_loader)
        # Initialized all data_loaders for all workers.

        # Initializing test_data_loader.
        if model_name not in ["bert", "gpt"]:
            test_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
            )
            test_dataset = datasets.CIFAR10(root=training_data_dir, train=False, download=True, transform=test_transform)
            test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
            # Initialized test_data_loader.
        else:
            # for bert and gpt loader
            test_dataset = None
            test_data_loader = torch.utils.data.DataLoader(test_ds, batch_size=2, drop_last=True)
        # Initialized test_data_loader.

        logging.info(f"{job_name} manager initializing.")
        rpc.init_rpc("manager", rank=0, world_size=world_size, rpc_backend_options=rpc_backend_options)
        logging.info(f"{job_name} manager initialized.")

        wrk_rrefs = []
        tester_rref = None

        for id in range(wrk_num):
            wrk_rref = rpc.remote(
                to=f"worker{id}",
                func=Worker,
                args=(
                    id,
                    job_name,
                    model_name,
                    model,
                    wrk_num,
                    epoch_num,
                    learning_rate,
                    gpu_ids[id + 1],
                    training_data_dir,
                    batch_size,
                    data_loaders[id],
                    wrk_pkey_lists,
                    wrk_pkey_locs,
                ),
            )
            wrk_rrefs.append(wrk_rref)
        tester_rref = rpc.remote(
            to="tester",
            func=Tester,
            args=(
                job_name,
                model_name,
                model,
                wrk_num,
                gpu_ids[-1],
                training_data_dir,
                wrk_rrefs,
                # model_fetch_ids,
                test_batch_size,
                test_target_loss,
                test_data_loader,
                test_dataset,
                test_wrk_id,
            ),
        )
        for wrk_rref in wrk_rrefs:
            wrk_rref.rpc_sync().set_ps_worker_rrefs_and_tester_rref(wrk_rrefs, tester_rref)

        futures = []
        for id in range(wrk_num):
            futures.append(rpc.rpc_async(to=f"worker{id}", func=Worker.run_worker, args=(wrk_rrefs[id],)))
        futures.append(rpc.rpc_async(to="tester", func=Tester.test_model, args=(tester_rref, test_df)))
        torch.futures.wait_all(futures)

        logging.info("All workers and tester complete.")
    elif rpc_rank <= wrk_num:  # workers
        logging.info(f"{job_name} worker{rpc_rank - 1} initializing.")
        rpc.init_rpc(f"worker{rpc_rank - 1}", rank=rpc_rank, world_size=world_size, rpc_backend_options=rpc_backend_options)
        logging.info(f"{job_name} worker{rpc_rank - 1} initialized.")
    else:  # tester
        logging.info(f"{job_name} tester initializing.")
        rpc.init_rpc("tester", rank=rpc_rank, world_size=world_size, rpc_backend_options=rpc_backend_options)
        logging.info(f"{job_name} tester initialized.")

    rpc.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--job_name", type=str, default="job0")
    parser.add_argument("--model_name", type=str, default="alexnet")
    parser.add_argument("--rpc_rank", type=int, default=0)
    parser.add_argument("--worker_num", type=int, default=1)
    parser.add_argument("--training_data_dir", type=str, default="./training_data/")
    parser.add_argument("--rpc_master_addr", type=str, default="localhost")
    parser.add_argument("--rpc_master_port", type=str, default="29600")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--test_target_loss", type=float, default=0.8)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--epoch_num", type=int, default=1)
    parser.add_argument("--data_partitioned", type=int, default=1)
    parser.add_argument("--gpu_ids", type=str, required=True)
    # parser.add_argument("--model_fetch_ids", type=str, default="0")

    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = args.rpc_master_addr
    os.environ["MASTER_PORT"] = args.rpc_master_port

    main(
        args.job_name,
        args.model_name,
        args.rpc_rank,
        args.worker_num,
        args.training_data_dir,
        args.batch_size,
        args.test_batch_size,
        args.test_target_loss,
        args.learning_rate,
        args.epoch_num,
        args.data_partitioned,
        args.gpu_ids.split(","),
        # args.model_fetch_ids.split(","),
    )
