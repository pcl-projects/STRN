#! /usr/bin/env python3


import argparse
import copy
import logging
import os
import random
import threading
import time

import numpy as np
import pandas as pd
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


class LoggerConstructor(object):
    def __init__(self, logger_name, file_name, log_level=logging.INFO, mode="w"):
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)
        self.__fh = logging.FileHandler(filename=file_name, mode=mode)
        self.__formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
        self.__fh.setFormatter(self.__formatter)
        self.__logger.addHandler(self.__fh)

    def get_logger(self):
        return self.__logger


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


class ParameterServer:
    def __init__(
        self,
        ps_id,
        job_name,
        model_name,
        ps_num,
        worker_num,
        batch_size,
        learning_rate,
        param_ps_idx,
        param_loc_idx,
        ps_params,
    ) -> None:
        self.ps_id = ps_id
        self.job_name = job_name
        self.model_name = model_name
        self.ps_num = ps_num
        self.worker_num = worker_num
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.param_ps_idx = param_ps_idx
        self.param_loc_idx = param_loc_idx
        self.ps_params = ps_params
        self.ps_rrefs = None
        self.worker_rrefs = None
        self.tester_rref = None
        self.logger = LoggerConstructor(
            logger_name=job_name, file_name=f"./training_logs/{job_name}_ps{ps_id}.log"
        ).get_logger()
        self.logger.info(f"Model name: {model_name}")

        self.ps_comp_future = torch.futures.Future()
        self.sync_mode = 0
        self.sync_lock = threading.Lock()
        self.sync_worker_count = 0

        for param in self.ps_params:
            param.grad = None
        # self.optimizer = optim.SGD(self.ps_params, lr=learning_rate, momentum=0.9)
        if self.model_name == "gpt":
            # self.optimizer = optim.Adam(self.ps_params, lr=1e-6)
            self.optimizer = optim.Adam(self.ps_params, lr=1e-6)
        elif self.model_name == "bert":
            self.optimizer = optim.Adam(self.ps_params, lr=1e-5)
        else:
            self.optimizer = optim.SGD(self.ps_params, lr=learning_rate, momentum=0.9)

        self.step_num = 0

        # For worker/PS computation time and communication time
        self.ps_cp_t = [0.0 for _ in range(worker_num)]
        self.wrk_cp_t = [0.0 for _ in range(worker_num)]
        self.in_cm_t = [0.0 for _ in range(worker_num)]
        self.out_cm_t = [0.0 for _ in range(worker_num)]

        self.epoch_iter_idx_count = [(0, 0, 0) for _ in range(worker_num)]

    def set_ps_worker_rrefs_and_tester_rref(self, ps_rrefs, worker_rrefs, tester_rref):
        if not isinstance(self, ParameterServer):
            self = self.local_value()

        self.ps_rrefs = ps_rrefs
        self.worker_rrefs = worker_rrefs
        self.tester_rref = tester_rref

    def set_sync_mode(self, sync_mode):
        if not isinstance(self, ParameterServer):
            self = self.local_value()

        self.sync_mode = sync_mode

    def get_step_num(self):
        if not isinstance(self, ParameterServer):
            self = self.local_value()

        return self.step_num

    def get_ps_params(self):
        if not isinstance(self, ParameterServer):
            self = self.local_value()

        return self.ps_params

    @rpc.functions.async_execution
    def ps_computing(self, worker_id, epoch_iter_idx_count, gradients, wrk_cp_t, ps2wrk_out_cm_t, in_cm_t_0):
        in_cm_t = 1000 * (time.time() - in_cm_t_0)

        if not isinstance(self, ParameterServer):
            self = self.local_value()

        with self.sync_lock:
            sync_mode = int(self.sync_mode)

            self.wrk_cp_t[worker_id] = wrk_cp_t
            self.in_cm_t[worker_id] = in_cm_t
            self.out_cm_t[worker_id] = ps2wrk_out_cm_t
            self.epoch_iter_idx_count[worker_id] = epoch_iter_idx_count

            epoch = epoch_iter_idx_count[0]
            batch_idx = epoch_iter_idx_count[1]

            self.logger.info(
                "Epoch: {:3d} | Batch: {:3d} | wrk: {} | Communication O: {:7.2f} ms".format(
                    epoch, batch_idx, worker_id, ps2wrk_out_cm_t
                )
            )
            self.logger.info(
                "Epoch: {:3d} | Batch: {:3d} | wrk: {} | Communication I: {:7.2f} ms".format(
                    epoch, batch_idx, worker_id, in_cm_t
                )
            )

            self.step_num += 1

            for param, grad in zip(self.ps_params, gradients):
                if param.grad is None:
                    param.grad = grad
                else:
                    param.grad += grad
            self.sync_worker_count += 1

            ft = self.ps_comp_future

            if sync_mode == 1 or self.sync_worker_count >= self.worker_num:
                ps_cp_t_0 = time.time()

                # if sync_mode == 0 or (sync_mode == 1 and self.sync_worker_count > 1):
                #     for param in self.ps_params:
                #         param.grad = param.grad / self.sync_worker_count
                # param.grad = param.grad / 4

                self.optimizer.step()
                # self.optimizer.zero_grad()
                for param in self.ps_params:
                    param.grad = None
                self.sync_worker_count = 0

                ps_cp_t = 1000 * (time.time() - ps_cp_t_0)

                if sync_mode == 0:
                    for i in range(self.worker_num):
                        self.ps_cp_t[i] = ps_cp_t
                else:
                    self.ps_cp_t[worker_id] = ps_cp_t

                self.logger.info("PS Comp Time {:.4f} ms".format(ps_cp_t))
                self.logger.info(f"PS sending paras to worker{worker_id}")

                ft.set_result((self.ps_params, time.time()))
                self.ps_comp_future = torch.futures.Future()

        return ft


class Worker:
    def __init__(
        self,
        worker_id,
        job_name,
        model_name,
        model,
        ps_num,
        worker_num,
        training_data_dir,
        batch_size,
        epoch_num,
        gpu_id,
        data_loader,
        grad_lists,
        param_lists,
        param_ps_idx,
        param_loc_idx,
    ) -> None:
        self.worker_id = worker_id
        self.job_name = job_name
        self.model_name = model_name
        self.model = model
        self.ps_num = ps_num
        self.worker_num = worker_num
        self.training_data_dir = training_data_dir
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.gpu_id = gpu_id
        self.data_loader = data_loader
        self.grad_lists = grad_lists
        self.param_lists = param_lists
        self.param_ps_idx = param_ps_idx
        self.param_loc_idx = param_loc_idx
        self.ps_rrefs = None
        self.worker_rrefs = None
        self.tester_rref = None
        self.logger = LoggerConstructor(
            logger_name=job_name, file_name=f"./training_logs/{job_name}_worker{worker_id}.log"
        ).get_logger()
        self.logger.info(f"Model name: {model_name}")

        self.epoch_idx = 0
        self.iter_idx = 0
        self.iter_count = 0

        self.out_cm_t = [0.0 for _ in range(ps_num)]

    def set_ps_worker_rrefs_and_tester_rref(self, ps_rrefs, worker_rrefs, tester_rref):
        if not isinstance(self, Worker):
            self = self.local_value()

        self.ps_rrefs = ps_rrefs
        self.worker_rrefs = worker_rrefs
        self.tester_rref = tester_rref

    def run_ps_computing(self, ps_id, wrk_cp_t, ps2wrk_out_cm_t):
        if not isinstance(self, Worker):
            self = self.local_value()

        ps_rref = self.ps_rrefs[ps_id]
        ps_params, out_cm_t_0 = ps_rref.rpc_sync().ps_computing(
            self.worker_id,
            (self.epoch_idx, self.iter_idx, self.iter_count),
            self.grad_lists[ps_id],
            wrk_cp_t,
            ps2wrk_out_cm_t,
            time.time(),
        )
        out_cm_t = 1000 * (time.time() - out_cm_t_0)
        self.out_cm_t[ps_id] = out_cm_t
        self.param_lists[ps_id] = ps_params

    def run_worker(self):
        if not isinstance(self, Worker):
            self = self.local_value()

        device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)

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

        self.model = self.model.to(device)
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

                if self.model_name == "lstm" or self.model_name == "transformer":
                    i = x_pack
                    data, target = get_batch(train_data, i, bptt)
                else:
                    data, target = x_pack

                wrk_cp_t_0 = time.time()

                if self.model_name == "bert" or self.model_name == "gpt":
                    target = target.to(device)
                    mask = data["attention_mask"].squeeze().to(device)
                    input_ids = data["input_ids"].squeeze().to(device)
                    self.model.zero_grad()
                    output = self.model(input_ids, mask)
                    batch_loss = criterion(output, target.long())
                    batch_loss.backward()
                else:
                    # other models
                    # self.model = self.model.to(device)
                    data, target = data.to(device), target.to(device)
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
                    # self.model = self.model.cpu()

                # torch.cuda.synchronize()

                for key, param in self.model.named_parameters():
                    self.grad_lists[self.param_ps_idx[key]][self.param_loc_idx[key]] = param.grad.cpu()

                wrk_cp_t = 1000 * (time.time() - wrk_cp_t_0)

                threads = []
                out_cm_t_copy = copy.deepcopy(self.out_cm_t)
                for ps_id in range(self.ps_num):
                    thrd = threading.Thread(target=self.run_ps_computing, args=(ps_id, wrk_cp_t, out_cm_t_copy[ps_id]))
                    thrd.start()
                    threads.append(thrd)
                for thrd in threads:
                    thrd.join()

                for key in self.param_ps_idx:
                    param = self.param_lists[self.param_ps_idx[key]][self.param_loc_idx[key]]  # .to(device)
                    self.model.state_dict()[key].copy_(param)

                self.logger.info(
                    "Epoch: {:3d} | Batch: {:3d} | Computation Time: {:7.2f} ms".format(self.epoch_idx, self.iter_idx, wrk_cp_t)
                )

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
        self.logger = LoggerConstructor(job_name=job_name, file_path=f"./training_logs/{job_name}_tester.log").logger
        self.logger.info(f"Model name: {model_name}")

        self.stop_flag = False

    def get_stop_flag(self):
        if not isinstance(self, Tester):
            self = self.local_value()

        return self.stop_flag

    def test_model(self, test_df=None):
        if not isinstance(self, Tester):
            self = self.local_value()

        device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)

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
        self.model = self.model.to(device)
        self.model.eval()
        while True:
            time_1 = time.time()

            if time_1 - time_0 >= 180:
                time_0 = time_1

                # fetch_index = self.cur_fetch_index
                # if fetch_index == len(self.model_fetch_ids) - 1:
                #     self.cur_fetch_index = 0
                # else:
                #     self.cur_fetch_index += 1
                futures = []
                for i in range(self.ps_num):
                    ps_rref = self.ps_rrefs[i]
                    futures.append(ps_rref.rpc_async().get_ps_params())
                ps_param_lists = []
                for future in futures:
                    ps_params = future.wait()
                    ps_param_lists.append(ps_params)
                # self.model = self.model.cpu()
                for key in self.param_ps_idx:
                    param = ps_param_lists[self.param_ps_idx[key]][self.param_loc_idx[key]]  # .to(device)
                    self.model.state_dict()[key].copy_(param)
                # self.model = self.model.to(device)

                test_correct = 0.0
                test_loss = 0.0

                if self.model_name in ["bert", "gpt"]:
                    total_acc_val = 0
                    total_loss_val = 0
                    with torch.no_grad():
                        for test_input, test_label in self.test_data_loader:
                            test_label = test_label.to(device)
                            mask = test_input["attention_mask"].squeeze().to(device)
                            input_id = test_input["input_ids"].squeeze().to(device)
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
                                data, targets = data.to(device), targets.to(device)
                                hidden = repackage_hidden(hidden)
                                output, hidden = self.model(data, hidden)
                                loss = criterion(output, targets)
                                test_loss += len(data) * loss.item()
                    elif self.model_name == "transformer":
                        with torch.no_grad():
                            for _, i in enumerate(range(0, val_data.size(0) - 1, bptt)):
                                data, targets = get_batch(val_data, i, bptt)
                                data, targets = data.to(device), targets.to(device)
                                output = self.model(data)
                                output = output.view(-1, ntokens)
                                loss = criterion(output, targets)
                                test_loss += len(data) * loss.item()
                    else:
                        with torch.no_grad():
                            for _, (data, target) in enumerate(self.test_data_loader):
                                data, target = data.to(device), target.to(device)
                                output = self.model(data)
                                loss = criterion(output, target)
                                test_loss += loss.item()
                                _, predicted = output.max(1)
                                test_correct += predicted.eq(target).sum().item()

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
    ps_num,
    worker_num,
    training_data_dir,
    batch_size,
    test_batch_size,
    test_target_loss,
    learning_rate,
    epoch_num,
    data_partitioned,
    gpu_ids,
    # model_fetch_ids,
):
    logging.basicConfig(level=logging.INFO)
    world_size = ps_num + worker_num + 2
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
        param_numel = {}
        key_param = {}
        ps_param_nums = [0 for _ in range(ps_num)]
        ps_param_lists = [[] for _ in range(ps_num)]  # needed for ps
        param_ps_idx = {}  # needed for ps and worker
        param_loc_idx = {}  # needed for ps and worker
        for key, param in model.named_parameters():
            param_numel[key] = param.numel()
            key_param[key] = param
        param_numel = sorted(param_numel.items(), key=lambda x: x[1], reverse=True)
        for i in range(len(param_numel)):
            key = param_numel[i][0]
            idx = np.argmin(ps_param_nums)
            param_ps_idx[key] = idx
            param_loc_idx[key] = len(ps_param_lists[idx])
            ps_param_nums[idx] += param_numel[i][1]
            ps_param_lists[idx].append(key_param[key])
        param_or_grad_lists = []
        for ps_params in ps_param_lists:
            param_or_grad_lists.append([None for _ in range(len(ps_params))])
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
                worker_dataset_len = int((dataset_len + worker_num) / worker_num)
                len_list = [worker_dataset_len for _ in range(worker_num - 1)]
                len_list.append(dataset_len - (worker_num - 1) * worker_dataset_len)
                training_datasets = random_split(training_dataset, len_list)
                for id in range(worker_num):
                    data_loader = DataLoader(training_datasets[id], batch_size=batch_size, shuffle=True)
                    data_loaders.append(data_loader)
            else:
                for _ in range(worker_num):
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
            wrk_ds_len = int((train_ds_len + worker_num) / worker_num)
            len_list = [wrk_ds_len for _ in range(worker_num - 1)]
            len_list.append(train_ds_len - (worker_num - 1) * wrk_ds_len)
            train_ds_list = torch.utils.data.random_split(train_ds, len_list)
            for idx in range(worker_num):
                data_loader = torch.utils.data.DataLoader(train_ds_list[idx], batch_size=2, shuffle=True, drop_last=True)
                data_loaders.append(data_loader)

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

        logging.info(f"{job_name} manager initializing.")
        rpc.init_rpc("manager", rank=0, world_size=world_size, rpc_backend_options=rpc_backend_options)
        logging.info(f"{job_name} manager initialized.")

        ps_rrefs = []
        worker_rrefs = []
        tester_rref = None

        for id in range(ps_num):
            ps_rref = rpc.remote(
                to=f"ps{id}",
                func=ParameterServer,
                args=(
                    id,
                    job_name,
                    model_name,
                    ps_num,
                    worker_num,
                    batch_size,
                    learning_rate,
                    param_ps_idx,
                    param_loc_idx,
                    ps_param_lists[id],
                ),
            )
            ps_rrefs.append(ps_rref)
        for id in range(worker_num):
            worker_rref = rpc.remote(
                to=f"worker{id}",
                func=Worker,
                args=(
                    id,
                    job_name,
                    model_name,
                    model,
                    ps_num,
                    worker_num,
                    training_data_dir,
                    batch_size,
                    epoch_num,
                    gpu_ids[id],
                    data_loaders[id],
                    param_or_grad_lists,
                    param_or_grad_lists,
                    param_ps_idx,
                    param_loc_idx,
                ),
            )
            worker_rrefs.append(worker_rref)
        tester_rref = rpc.remote(
            to="tester",
            func=Tester,
            args=(
                job_name,
                model_name,
                model,
                ps_num,
                worker_num,
                training_data_dir,
                gpu_ids[-1],
                ps_rrefs,
                worker_rrefs,
                # model_fetch_ids,
                test_batch_size,
                test_target_loss,
                test_data_loader,
                test_dataset,
                param_ps_idx,
                param_loc_idx,
            ),
        )
        for ps_rref in ps_rrefs:
            ps_rref.rpc_sync().set_ps_worker_rrefs_and_tester_rref(ps_rrefs, worker_rrefs, tester_rref)
        for worker_rref in worker_rrefs:
            worker_rref.rpc_sync().set_ps_worker_rrefs_and_tester_rref(ps_rrefs, worker_rrefs, tester_rref)

        futures = []
        for id in range(worker_num):
            futures.append(rpc.rpc_async(to=f"worker{id}", func=Worker.run_worker, args=(worker_rrefs[id],)))
        futures.append(rpc.rpc_async(to="tester", func=Tester.test_model, args=(tester_rref, test_df)))
        torch.futures.wait_all(futures)

        logging.info("All workers and tester complete.")
    elif rpc_rank <= ps_num:  # ps-s
        logging.info(f"{job_name} ps{rpc_rank - 1} initializing.")
        rpc.init_rpc(f"ps{rpc_rank - 1}", rank=rpc_rank, world_size=world_size, rpc_backend_options=rpc_backend_options)
        logging.info(f"{job_name} ps{rpc_rank - 1} initialized.")
    elif rpc_rank <= ps_num + worker_num:  # workers
        logging.info(f"{job_name} worker{rpc_rank - ps_num - 1} initializing.")
        rpc.init_rpc(
            f"worker{rpc_rank - ps_num - 1}", rank=rpc_rank, world_size=world_size, rpc_backend_options=rpc_backend_options
        )
        logging.info(f"{job_name} worker{rpc_rank - ps_num - 1} initialized.")
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
    parser.add_argument("--ps_num", type=int, default=1)
    parser.add_argument("--worker_num", type=int, default=1)
    parser.add_argument("--training_data_dir", type=str, default="./training_data/")
    parser.add_argument("--rpc_master_addr", type=str, default="localhost")
    parser.add_argument("--rpc_master_port", type=str, default="29600")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--test_target_loss", type=float, default=0.8)
    parser.add_argument("--learning_rate", type=float, default=0.005)
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
        args.ps_num,
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
