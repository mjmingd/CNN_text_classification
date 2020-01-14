import argparse
import json
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data import SampleDataset, WordProcessor
from model import WordShallowNet
from utils import init_weights, acc, save_model
import build_vocab

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../dataset/')
parser.add_argument('--pos_file', default='rt-polarity.pos')
parser.add_argument('--neg_file', default='rt-polarity.neg')
parser.add_argument('--val_dir', default=None)
parser.add_argument('--val_pos_file', default=None)
parser.add_argument('--val_neg_file', default=None)
parser.add_argument('--model_dir', default='./model/')
parser.add_argument('--num_class', default=2, type=int)
parser.add_argument('--num_per_filters', default=100, type=int)
parser.add_argument('--vocab', default='vocab.pkl')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--seed', default=10, type=int)
parser.add_argument('--learning_rate', default=0.001, type=int)
parser.add_argument('--epochs', default=1, type=int)



def train_(model, train_dl, device, nEpoch, model_dir, checkpoint_name):

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

            if step % 1 == 0 :
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
                writer.add_scalars('loss', {'train': train_loss / (step + 1), 'val': val_summary['loss']}, epoch + 1)
                print('val_loss: {:.3f}, val_acc: {:.2%}'.format(val_summary['loss'],
                                                                                 val_summary['acc']))

    state = {'epoch': args.epochs + 1,
                 'model_state_dict': net.state_dict(),
                 'opt_state_dict': optimizer.state_dict()}

    save_model(state, model_dir + checkpoint_name + '.tar')

def evaluate(model, data_loader, metrics, device):
    if model.training:
        model.eval()

    summary = {metric: 0 for metric in metrics}

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

    if args.vocab is None :
        build_vocab.main()

    with open(args.data_dir + args.vocab, mode='rb') as io:
        vocab = pickle.load(io)

    with open(args.data_dir + 'vocab_config.json', 'r') as f:
        vocab_config = json.loads(f.read())

    word_tk = WordProcessor(vocab, vocab_config['max_seq_len'])
    train_ds = SampleDataset(args.data_dir, args.pos_file, args.neg_file, word_tk.transform)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    if args.val_dir is not None:
        val_ds = SampleDataset(args.val_dir, args.val_pos_file, args.val_neg_file, word_tk.transform)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    net = WordShallowNet(word_tk, args.num_class, vocab_config['max_seq_len'], args.num_per_filters)


    net.apply(init_weights)
    torch.manual_seed(args.seed)

    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    writer = SummaryWriter('{}/runs'.format(args.model_dir))
    checkpoint_name = 'word_Densenet'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    train_(net, train_dl, device, args.epochs, args.model_dir, checkpoint_name)
