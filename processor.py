import os
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer

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
    # tokens: [["中", "国", "的", "首", "都", "是", "北", "京"], [...], ...]
    input_ids = []
    for t in tokens:
        tokenized_text = tokenizer.encode(t, add_special_tokens=False, max_length=512)
        input_ids.append(tokenized_text)
    return input_ids


def sequence_padding(tokenizer, labels, X, y, max_len=128, add_CLS=True):
    X = convert_tokens_to_ids(tokenizer, X)
    y = [[labels.index(j) for j in i ] for i in y]
    
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    cls_id = tokenizer.convert_tokens_to_ids("[CLS]")

    tensorX = []
    tensory = []
    for i in range(len(X)):
        curX, cury = [], []
        if len(X[i]) <= max_len:
            curX = X[i] + [pad_id] * (max_len - len(X[i]))
            cury = y[i] + [-100] * (max_len - len(y[i]))
        else:
            curX = X[i][:max_len]
            cury = y[i][:max_len]
        if add_CLS:
            curX = [cls_id] + curX[:max_len-1]
            cury = [-100] + cury[:max_len-1]
        tensorX.append(curX)
        tensory.append(cury)

    tensorX = torch.LongTensor(tensorX)
    tensory = torch.LongTensor(tensory)
    tensorMask = (tensorX.ne(pad_id) ).byte()
    return tensorX, tensorMask, tensory


class Tokenizer:
    def __init__(self, file_path):
        self.data, _ = read_file(file_path)
        self.dictionary = self.__build_dict__()
    
    def __build_dict__(self):
        print("Building dictionary...")
        dictionary = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3
        }
        for sen in self.data:
            for word in sen:
                dictionary[word] = dictionary.get(word, len(dictionary))
        print("Dictionary size is ", len(dictionary))
        return dictionary
    
    def convert_tokens_to_ids(self, token):
        return self.dictionary.get(token, self.dictionary["[UNK]"])
    
    def __len__(self):
        return len(self.dictionary)

    def encode_plus(self, t, add_special_tokens=False):
        tokens_t = []
        for token in t:
            tokens_t.append(self.convert_tokens_to_ids(token))
        if add_special_tokens:
            tokens_t = [self.convert_tokens_to_ids("[CLS]")] + tokens_t + [self.convert_tokens_to_ids("[SEP]")]
        return {
            "input_ids": tokens_t
        }

class NERDataset(Dataset):
    def __init__(self, file_path, tokenizer, labels, max_len=128, add_CLS=True):
        self.X, self.y = read_file(file_path)
        # a = set()
        # for i in self.y:
        #     a = a | set(i)
        # print(a)
        self.X, self.mask, self.y = sequence_padding(tokenizer, labels, self.X, self.y, max_len=max_len, add_CLS=add_CLS)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx], self.y[idx]