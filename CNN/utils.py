import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Embedding):
        m.weight.data.uniform_(-1.0, 1.0)
    if isinstance(m, nn.Conv2d):
        m.weight.data.uniform_(-1.0, 1.0)
        m.bias.data.fill_(0)

def acc(yhat, y):
    with torch.no_grad():
        yhat = yhat.max(dim=1)[1]
        acc = (yhat == y).float().mean()
    return acc

def save_model(state, filepath):
    torch.save(state, filepath)
