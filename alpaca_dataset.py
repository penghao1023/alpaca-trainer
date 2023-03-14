import torch
import torch.utils.data as data


class AlpacaData(torch.utils.data.Dataset):
    def __init__(self, tokenizer, max_len, subset):
        self.tokenizer = tokenizer
        self.subset = subset

    def __len__(self):
        return self.questions

    def __getitem__(self, index):
        batch = self.subset[index]


def split_train_val(train_set, tokenizer):
    train_set = train_set.map(print)
    train, val, test = data.random_split(train_set, [0.8, 0.1, 0.1])
    return train, val, test
