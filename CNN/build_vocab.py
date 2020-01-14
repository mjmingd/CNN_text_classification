from collections import Counter
import itertools
import gluonnlp as nlp
import re
import json
import pickle


def cleanSent(sent):
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

def load_data(filepath, pos_file, neg_file) :
    with open(filepath + pos_file, 'r', encoding='utf-8') as pf:
        pos_data = pf.readlines()
        pos_data = [cleanSent(s.lower()) for s in pos_data]
    with open(filepath + neg_file, 'r', encoding='utf-8') as nf:
        neg_data = nf.readlines()
        neg_data = [cleanSent(s.lower()) for s in neg_data]

    data = pos_data + neg_data

    return data

def tokenize(data) :
    tokens = [x.split() for x in data]
    max_seq_len = 0
    for x in tokens :
        if len(x) > max_seq_len :
            max_seq_len = len(x)
    return tokens, max_seq_len


def save_vocab(vocab, data_dir, vocab_name):
    with open(data_dir + vocab_name, mode='wb') as io:
        pickle.dump(vocab, io)


def main():
    data_dir = '../dataset/'
    pos_file = 'rt-polarity.pos'
    neg_file = 'rt-polarity.neg'
    vocab_name = 'vocab.pkl'
    min_freq = 5

    data = load_data(data_dir, pos_file, neg_file)
    tokens, max_seq_len = tokenize(data)
    tokens_counter = Counter(itertools.chain.from_iterable(tokens))
    tokens_list = [token_count[0] for token_count in tokens_counter.items() if token_count[1] >= min_freq]
    tokens_list = sorted(tokens_list)
    tokens_list = ['<unk>', '<pad>'] + tokens_list

    tmp_vocab = nlp.Vocab(counter=Counter(tokens_list), min_freq=1, bos_token=None, eos_token=None)
    ptr_embedding = nlp.embedding.create('fasttext', source='wiki.en')
    tmp_vocab.set_embedding(ptr_embedding)
    array = tmp_vocab.embedding.idx_to_vec.asnumpy()

    save_vocab(tmp_vocab, data_dir, vocab_name)

    vocab_config = {'vocab_size' : len(vocab), 'max_seq_len' : max_seq_len, 'embedding_dim' : 300,
                    'min_freq' : 5}

    with open(data_dir + 'vocab_config.json', 'w') as f :
        json.dump(vocab_config, f)

if __name__ == '__main__':
    main()
