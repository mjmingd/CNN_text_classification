import os
import sys
import re
import numpy as np
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchsummary

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./dataset/')
parser.add_argument('--pos_file', default='rt-polarity.pos')
parser.add_argument('--neg_file', default='rt-polarity.neg')
parser.add_argument('--val_dir', default=None)
parser.add_argument('--val_pos_file', default=None)
parser.add_argument('--val_neg_file', default=None)
parser.add_argument('--model_dir', default='./model/')
parser.add_argument('--num_class', default=2)
parser.add_argument('--mode', default='char')
parser.add_argument('--max_seq_len', default=1014)
parser.add_argument('--batch_size', default=128)
parser.add_argument('--seed', default=10)
parser.add_argument('--learning_rate', default=0.001)
parser.add_argument('--epochs', default=1)


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


class CharProcessor():
    def __init__(self):

        self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
        self.max_seq_len = args.max_seq_len
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


class CharNet(nn.Module):
    def __init__(self, vocab_size, num_class):
        super(CharNet, self).__init__()

        # DenseNet
        self.tempConv1 = nn.Conv2d(vocab_size, 64, 3, stride=1, padding=1)  # [vocab_size, 3]
        self.block1 = self.build_DenseBlock(64, 64, 4)
        self.trans1 = TransitionLayer(64, 128)

        self.block2 = self.build_DenseBlock(128, 128, 4)
        self.trans2 = TransitionLayer(128, 256)

        self.block3 = self.build_DenseBlock(256, 256, 4)
        self.trans3 = TransitionLayer(256, 512)

        self.block4 = self.build_DenseBlock(512, 512, 4)

        self.lastPool = nn.MaxPool2d([70, 1], 8)
        self.fc1 = nn.Linear(4096 * 2, 2048)
        self.fc2 = nn.Linear(2048, num_class)

    def build_DenseBlock(self, in_channels, out_channels, nBlocks=4):
        layers = []
        inter_channels = in_channels
        growth = out_channels // nBlocks
        for i in range(nBlocks):
            layers.append(ConvBlock(inter_channels, growth))
            inter_channels += growth
        return nn.Sequential(*layers)

    def forward(self, x):
        # torch.Size([1, 69, 1014, 1])
        x = self.tempConv1(x)  # torch.Size([1, 64, 1014, 1])
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.trans3(self.block3(x))
        x = self.block4(x)
        x = self.lastPool(x)
        x = x.view(-1, 4096 * 2)
        x = nn.functional.relu((self.fc1(x)))
        out = self.fc2(x)

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.BN = nn.BatchNorm2d(in_channels)
        self.Conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.Conv(self.relu(self.BN(x)))
        out = torch.cat((x, out), 1)
        return out


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.tempConv = nn.Conv2d(2 * in_channels, out_channels, (1, 3), stride=1, padding=1)
        self.maxPool = nn.MaxPool2d((2, 1), 2)

    def forward(self, x):
        out = self.tempConv(x)
        out = self.maxPool(out)

        return out

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.uniform_(-1.0, 1.0)
        m.bias.data.fill_(0)

def acc(yhat, y):
    with torch.no_grad():
        yhat = yhat.max(dim=1)[1]
        acc = (yhat == y).float().mean()
    return acc


def save_checkpoint(state, filename, model_dir):
    torch.save(state, model_dir + filename)


def train_(model, train_dl, device, nEpoch):

    for epoch in range(nEpoch):

        train_loss = 0
        train_acc = 0

        model.train()
        for step, mb in enumerate(train_dl):
            x_mb, y_mb = map(lambda elm: elm.to(device), mb)

            optimizer.zero_grad()
            y_hat_mb = model(x_mb)

            mb_loss = loss_fn(y_hat_mb, y_mb)
            mb_loss.backward()

            optimizer.step()

            with torch.no_grad():
                mb_acc = acc(y_hat_mb, y_mb)

            train_loss += mb_loss.item()
            train_acc += mb_acc.item()

            if step % 10 == 0 :
                print('epoch : {}, step : {}, train_loss: {:.3f}, train_acc: {:.2%}'
                           .format(epoch+1, step + 1, mb_loss, mb_acc))




        else:
            train_loss /= (step + 1)
            train_acc /= (step + 1)

            train_summary = {'loss': train_loss, 'acc': train_acc}

            print('epoch : {}, train_loss: {:.3f}, train_acc: {:.2%}'.format(epoch + 1, train_summary['loss'],
                                                                            train_summary['acc']))
            if args.val_dir is not None :
                val_summary = evaluate(model, val_dl, {'loss': loss_fn}, device)
                writer.add_scalars('loss', {'train': train_loss / (step + 1), 'val': val_loss}, epoch + 1)
                print('val_loss: {:.3f}, val_acc: {:.2%}'.format(val_summary['loss'],
                                                                                 val_summary['acc']))

def evaluate(model, data_loader, metrics, device):
    if model.training:
        model.eval()

    for step, mb in enumerate(data_loader):
        x_mb, y_mb = map(lambda elm: elm.to(device), mb)

        with torch.no_grad():
            y_hat_mb = model(x_mb)

            for metric in metrics:
                summary[metric] += metrics[metric](y_hat_mb, y_mb).item() * y_mb.size()[0]
    else:
        for metric in metrics:
            summary[metric] /= len(data_loader.dataset)

    return summary


if __name__ == '__main__':
    args = parser.parse_args()

    if args.mode == 'char' :
        char_tk = CharProcessor()
        train_ds = SampleDataset(args.data_dir, args.pos_file, args.neg_file, char_tk.transform)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

        if args.val_dir is not None :
            val_ds = SampleDataset(args.val_dir, args.val_pos_file, args.val_neg_file, char_tk.transform)
            val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

        net = CharNet(char_tk.vocab_size, args.num_class)


    net.apply(init_weights)
    torch.manual_seed(args.seed)
    # torchsummary.summary(net, (69,1014,1))

    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    writer = SummaryWriter('{}/runs'.format(args.model_dir))
    checkpoint = args.model_dir
    checkpoin_name = 'char_Densenet'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    train_(net, train_dl, device, args.epochs)

    state = {'epoch': args.epochs + 1,
             'model_state_dict': net.state_dict(),
             'opt_state_dict': optimizer.state_dict()}

    save_checkpoint(state, checkpoint_name + '.tar')


