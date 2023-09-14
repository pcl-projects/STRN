#! /usr/bin/env python3


"""https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f"""


import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

datapath = "./bbc-text.csv"
df = pd.read_csv(datapath)
# df.head()

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
labels = {"business": 0, "entertainment": 1, "sport": 2, "tech": 3, "politics": 4}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [labels[label] for label in df["category"]]
        self.texts = [
            tokenizer(text, padding="max_length", max_length=512, truncation=True, return_tensors="pt") for text in df["text"]
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


np.random.seed(112)
df_train, df_test = np.split(
    df.sample(frac=1, random_state=42),
    [
        int(0.8 * len(df)),
    ],
)


def train(model, train_data, test_data, learning_rate, epochs, batch_size):
    train, test = Dataset(train_data), Dataset(test_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input["attention_mask"].squeeze().to(device)
            input_id = train_input["input_ids"].squeeze().to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item() * len(train_label)

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for test_input, test_label in tqdm(test_dataloader):
                test_label = test_label.to(device)
                mask = test_input["attention_mask"].squeeze().to(device)
                input_id = test_input["input_ids"].squeeze().to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, test_label.long())
                total_loss_val += batch_loss.item() * len(test_label)

                acc = (output.argmax(dim=1) == test_label).sum().item()
                total_acc_val += acc

        print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Test Loss: {total_loss_val / len(test_data): .3f} \
                | Test Accuracy: {total_acc_val / len(test_data): .3f}"
        )


EPOCHS = 5
model = BertClassifier()
print(type(model))
LR = 1e-6
BATCH_SIZE = 2

train(model, df_train, df_test, LR, EPOCHS, BATCH_SIZE)
