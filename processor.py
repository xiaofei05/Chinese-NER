import os
from torch.utils.data import Dataset
import torch

def read_file(filename):
    X, y = [], []
    labels = []
    with open(filename, 'r', encoding='utf-8') as f:
        x0, y0 = [], []
        for line in f:
            data = line.strip()
            if data:
                x0.append(data.split()[0])
                y0.append(data.split()[1])
            else:
                X.append(x0)
                y.append(y0)
                x0, y0 = [], []
        if len(x0)!=0:
            X.append(x0)
            y.append(y0)
    return X, y


def convert_tokens_to_ids(tokenizer, tokens):
    input_ids = []
    cls_id = tokenizer.convert_tokens_to_ids("[CLS]")
    for t in tokens:
        tokenized_text = tokenizer.encode_plus(t, add_special_tokens=False)
        input_ids.append([cls_id] + tokenized_text["input_ids"])
    return input_ids


def sequence_padding(tokenizer, X, y, max_len=128):
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    y = [[-100] + i for i in y]
    tensorX = []
    tensory = []
    for i in range(len(X)):
        if len(X[i]) < max_len:
            tensorX.append(X[i] + [pad_id] * (max_len - len(X[i])))
            tensory.append(y[i] + [-100] * (max_len - len(y[i])))
        else:
            tensorX.append(X[i][:max_len])
            tensory.append(y[i][:max_len])
    tensorX = torch.LongTensor(tensorX)
    tensory = torch.LongTensor(tensory)
    return tensorX, tensory



class NERDataset(Dataset):
    def __init__(self, file_path, tokenizer, labels, max_len=128):
        self.X, self.y = read_file(file_path)
        self.X = convert_tokens_to_ids(tokenizer, self.X)
        self.y = [[labels.index(j) for j in i ] for i in self.y]
        self.X, self.y = sequence_padding(tokenizer, self.X, self.y, max_len=max_len)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    