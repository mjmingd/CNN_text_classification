import re
import numpy as np
from torch.utils.data import Dataset
import torch

class SampleDataset(Dataset):
    def __init__(self, filepath, pos_file, neg_file, transform_fn):
        """
        Args:
            filepath (str): filepath
            transform_fn (Callable): a function that can act as a transformer
        """
        with open(filepath + pos_file, 'r', encoding='utf-8') as pf:
            pos_data = pf.readlines()
            pos_data = [self.cleanSent(s.lower()) for s in pos_data]
        with open(filepath + neg_file, 'r', encoding='utf-8') as nf:
            neg_data = nf.readlines()
            neg_data = [self.cleanSent(s.lower()) for s in neg_data]

        self.data = pos_data + neg_data
        self.labels = np.concatenate([[1 for _ in pos_data], [0 for _ in neg_data]])
        self._transform = transform_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tensors = self._transform(self.data[idx])
        label = torch.tensor(self.labels[idx])
        return tensors, label

    def cleanSent(self, sent):
        sent = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sent)
        sent = re.sub(r"\'s", " \'s", sent)
        sent = re.sub(r"\'ve", " \'ve", sent)
        sent = re.sub(r"n\'t", " n\'t", sent)
        sent = re.sub(r"\'re", " \'re", sent)
        sent = re.sub(r"\'d", " \'d", sent)
        sent = re.sub(r"\'ll", " \'ll", sent)
        sent = re.sub(r",", " , ", sent)
        sent = re.sub(r"!", " ! ", sent)
        sent = re.sub(r"\(", " \( ", sent)
        sent = re.sub(r"\)", " \) ", sent)
        sent = re.sub(r"\?", " \? ", sent)
        sent = re.sub(r"\s{2,}", " ", sent)
        return sent.strip().lower()


class CharProcessor:
    def __init__(self, max_seq_len):

        self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
        self.max_seq_len = max_seq_len
        self.vocab_size = len(self.alphabet)

    def __len__(self):
        return len()

    def pad_seq(self, sent, pad_char=' '):
        if len(sent) < self.max_seq_len:
            pad = [pad_char] * (self.max_seq_len - len(sent))
            sent = sent + pad
        else:
            sent = sent[:self.max_seq_len]
        return sent

    def char2idx(self, s):
        return self.alphabet.find(s)

    def char2tensor(self, char):
        tensor = torch.zeros(1, self.vocab_size)
        tensor[0][self.char2idx(char)] = 1
        return tensor

    def sent2tensor(self, sent):
        tensor = torch.zeros(self.vocab_size, self.max_seq_len, 1)
        for li, s in enumerate(sent):
            tensor[self.char2idx(s)][li][0] = 1
        return tensor

    def transform(self, sent):
        return self.sent2tensor(self.pad_seq(list(sent)))
