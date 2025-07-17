from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import torch

tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for label, line in data_iter:
        yield tokenizer(line)

def build_dataset(batch_size=32, max_len=200):
    train_iter, test_iter = IMDB()
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])

    def process(example):
        label, line = example
        label = 1 if label == "pos" else 0
        tokens = tokenizer(line)
        tokens = tokens[:max_len]
        ids = vocab(tokens)
        ids += [vocab["<pad>"]] * (max_len - len(ids))
        return torch.tensor(ids), torch.tensor(label)

    train_iter, test_iter = IMDB()
    train_dataset = list(map(process, train_iter))
    test_dataset = list(map(process, test_iter))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, vocab