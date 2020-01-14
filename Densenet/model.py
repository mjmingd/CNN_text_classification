import torch
import torch.nn as nn


class CharNet(nn.Module):
    def __init__(self, vocab_size, num_class):
        super(CharNet, self).__init__()

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
        self.tempConv = nn.Conv2d(in_channels, out_channels, (1, 3), stride=1, padding=1)
        self.maxPool = nn.MaxPool2d((2, 1), 2)

    def forward(self, x):
        out = self.tempConv(x)
        out = self.maxPool(out)

        return out