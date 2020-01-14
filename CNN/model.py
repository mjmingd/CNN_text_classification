import torch
import torch.nn as nn
import torch.nn.functional as F


class CharShallowNet(nn.Module):
    def __init__(self, vocab_size, num_class, num_per_filters=700):
        super(CharShallowNet, self).__init__()
        self.nFilters = num_per_filters
        self.vocab_size = vocab_size

        # Shallow-Wide-net
        self.Conv1 = nn.Conv2d(self.vocab_size, self.nFilters, (1, 15))
        self.Conv2 = nn.Conv2d(self.vocab_size, self.nFilters, (1, 20))
        self.Conv3 = nn.Conv2d(self.vocab_size, self.nFilters, (1, 25))

        self._dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.nFilters * 3, num_class)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        x1 = F.relu(self.Conv1(x))
        x2 = F.relu(self.Conv2(x))
        x3 = F.relu(self.Conv3(x))
        out = torch.cat([x1.max(dim=-1)[0], x2.max(dim=-1)[0], x3.max(dim=-1)[0]], dim=-1)
        out = self._dropout(out)
        out = out.view(-1, self.nFilters * 3)
        out = self.fc(out)

        return out


class CharDenseNet(nn.Module):
    def __init__(self, vocab_size, num_class):
        super(CharDenseNet, self).__init__()

        # DenseNet
        self.tempConv1 = nn.Conv2d(vocab_size, 64, 3, stride=1, padding=1)  # [vocab_size, 3]
        self.block1 = self.build_DenseBlock(64, 64, 4)
        self.bc1 = nn.Conv2d(128, 64, 1, stride=1)
        self.trans1 = TransitionLayer(64, 128)

        self.block2 = self.build_DenseBlock(128, 128, 4)
        self.bc2 = nn.Conv2d(256, 128, 1, stride=1)
        self.trans2 = TransitionLayer(128, 256)

        self.block3 = self.build_DenseBlock(256, 256, 4)
        self.bc3 = nn.Conv2d(512, 256, 1, stride=1)
        self.trans3 = TransitionLayer(256, 512)

        self.block4 = self.build_DenseBlock(512, 512, 4)
        self.bc4 = nn.Conv2d(1024, 512, 1, stride=1)
        self.pool = nn.MaxPool2d((2, 1), 2)

        self.lastPool = nn.MaxPool2d(1, 8)
        self.fc1 = nn.Linear(4096, 2048)
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
        x = self.trans1(self.bc1(self.block1(x)))
        x = self.trans2(self.bc2(self.block2(x)))
        x = self.trans3(self.bc3(self.block3(x)))
        x = self.pool(self.bc4(self.block4(x)))
        x = self.lastPool(x)
        x = x.view(-1, 4096)
        x = F.relu((self.fc1(x)))
        out = self.fc2(x)

        return out

class WordShallowNet(nn.Module):
    def __init__(self, word_tk, num_class, max_seq_len, num_per_filters = 100, embedding_dim=300):
        super(WordShallowNet, self).__init__()
        self.emb_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.nFilters = num_per_filters
        self.emb_static = nn.Embedding.from_pretrained(torch.from_numpy(word_tk.embedding),
                                                       freeze=True, padding_idx=word_tk.vocab.to_indices(
                word_tk.vocab.padding_token))
        self.emb_non_static = nn.Embedding.from_pretrained(torch.from_numpy(word_tk.embedding),
                                                           freeze=False, padding_idx=word_tk.vocab.to_indices(
                word_tk.vocab.padding_token))

        # Shallow-Wide-net
        self.Conv1 = nn.Conv2d(self.max_seq_len, self.nFilters, (1, 3))
        self.Conv2 = nn.Conv2d(self.max_seq_len, self.nFilters, (1, 4))
        self.Conv3 = nn.Conv2d(self.max_seq_len, self.nFilters, (1, 5))

        self._dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.nFilters * 3, num_class)

    def forward(self, x):
        static = self.emb_static(x).view(-1, self.max_seq_len, self.emb_dim, 1).permute(0, 1, 3, 2)
        non_static = self.emb_non_static(x).view(-1, self.max_seq_len, self.emb_dim, 1).permute(0, 1, 3, 2)
        x1 = F.relu(self.Conv1(static)) + F.relu(self.Conv1(non_static))
        x2 = F.relu(self.Conv2(static)) + F.relu(self.Conv2(non_static))
        x3 = F.relu(self.Conv3(static)) + F.relu(self.Conv3(non_static))
        out = torch.cat([x1.max(dim=-1)[0], x2.max(dim=-1)[0], x3.max(dim=-1)[0]], dim=-1)
        out = self._dropout(out)
        out = out.view(-1, self.nFilters * 3)
        out = self.fc(out)

        return out



class WordDenseNet(nn.Module):
    def __init__(self, word_tk, num_class, max_seq_len, embedding_dim=300):
        super(WordDenseNet, self).__init__()

        self.emb_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding.from_pretrained(torch.from_numpy(word_tk.embedding),
                                                freeze=False, padding_idx=word_tk.vocab.to_indices(word_tk.vocab.padding_token))

        # DenseNet
        self.tempConv1 = nn.Conv2d(self.max_seq_len, 64, 3, stride=1, padding=1)
        self.block1 = self.build_DenseBlock(64, 64, 10)
        self.bc1 = nn.Conv2d(128, 64, 1, stride=1)
        self.trans1 = TransitionLayer(64, 128)

        self.block2 = self.build_DenseBlock(128, 128, 10)
        self.bc2 = nn.Conv2d(256, 128, 1, stride=1)
        self.trans2 = TransitionLayer(128, 256)

        self.block3 = self.build_DenseBlock(256, 256, 4)
        self.bc3 = nn.Conv2d(512, 256, 1, stride=1)
        self.trans3 = TransitionLayer(256, 512)

        self.block4 = self.build_DenseBlock(512, 512, 4)
        self.bc4 = nn.Conv2d(1024, 512, 1, stride=1)
        self.pool = nn.MaxPool2d((8, 1), 2)

        self.globalPool = nn.AdaptiveAvgPool2d(num_class)

    def build_DenseBlock(self, in_channels, out_channels, nBlocks=4):
        layers = []
        inter_channels = in_channels
        growth = out_channels // nBlocks
        for i in range(nBlocks):
            if i == nBlocks - 1 and inter_channels != out_channels:
                growth = in_channels + out_channels - inter_channels
            layers.append(ConvBlock(inter_channels, growth))
            inter_channels += growth

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.emb(x)
        x = x.view(-1, self.max_seq_len, self.emb_dim, 1)
        x = self.tempConv1(x)
        x = self.trans1(self.bc1(self.block1(x)))
        x = self.trans2(self.bc2(self.block2(x)))
        x = self.trans3(self.bc3(self.block3(x)))
        x = self.pool(self.bc4(self.block4(x)))
        x = self.globalPool(x)
        out = x.view(x.size()[0],-1)

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
        self.tempConv = nn.Conv2d(in_channels, out_channels, (1, 3), stride=1, padding=1)
        self.maxPool = nn.MaxPool2d((2, 1), 2)

    def forward(self, x):
        out = self.tempConv(x)
        out = self.maxPool(out)

        return out