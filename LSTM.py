import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class LSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, num_layers = 2, embSize = 128, dropout = 0.5, bidirectional = True):
        super(LSTM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embSize)
        self.lstm = nn.LSTM(embSize, hidden_size, num_layers, dropout = dropout, batch_first=True, bidirectional= bidirectional)
        self.fc = nn.Linear(hidden_size*2, output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)
        
        out = self.embedding(x)
        out, _ = self.lstm(out, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out)
        return out

class PreprocessorLSTM():
    def __init__(self, train_set, val_set, test_set ):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

        self.word_to_idx = {'<pad>': 0, '<unk>': 1}
        self.idx_to_word = {0: '<pad>', 1: '<unk>'}
        self.tag_to_idx = {'<pad>': 0, '<unk>': 1}
        self.idx_to_tag = {0: '<pad>', 1: '<unk>'}
        
        self.train_sentences = []
        self.train_labels = []
        self.dev_sentences = []
        self.dev_labels = []
        self.test_sentences = []
        self.test_labels = []

        self._get_sentences()
        self._get_vocab()

    def _word_cleaner(self, word):
        if word == '\'m':
            return 'am'
        elif word == '\'re':
            return 'are'
        elif word == 'n\'t':
            return 'not'
        elif word == '\'s':
            return 'is'
        elif word == '\'ve':
            return 'have'
        elif word == '\'ll':
            return 'will'
        elif word == '\'d':
            return 'would'    
        return word
    
    def _get_sentences(self):
        for sentence in self.train_set:
            self.train_sentences.append(['<sos>'] + [self._word_cleaner(word['form'].lower()) for word in sentence] + ['<eos>'])
            self.train_labels.append(['<pad>'] + [word['upostag'] for word in sentence] + ['<pad>'])

        for sentence in self.val_set:
            self.dev_sentences.append(['<sos>'] + [self._word_cleaner(word['form'].lower()) for word in sentence] + ['<eos>'])
            self.dev_labels.append(['<pad>'] + [word['upostag'] for word in sentence] + ['<pad>'])

        for sentence in self.test_set:
            self.test_sentences.append(['<sos>'] + [self._word_cleaner(word['form'].lower()) for word in sentence] + ['<eos>'])
            self.test_labels.append(['<pad>'] + [word['upostag'] for word in sentence] + ['<pad>'])

    def _get_vocab(self):
        for sentence in self.train_sentences:
            for word in sentence:
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = len(self.word_to_idx)
                    self.idx_to_word[len(self.idx_to_word)] = word
                
        for sentence in self.train_labels:
            for tag in sentence:
                if tag not in self.tag_to_idx:
                    self.tag_to_idx[tag] = len(self.tag_to_idx)
                    self.idx_to_tag[len(self.idx_to_tag)] = tag

        self.train_sentences = [[self.word_to_idx[word] if word in self.word_to_idx else self.word_to_idx['<unk>'] for word in sentence] for sentence in self.train_sentences]
        self.dev_sentences = [[self.word_to_idx[word] if word in self.word_to_idx else self.word_to_idx['<unk>'] for word in sentence] for sentence in self.dev_sentences]
        self.test_sentences = [[self.word_to_idx[word] if word in self.word_to_idx else self.word_to_idx['<unk>'] for word in sentence] for sentence in self.test_sentences]

        self.train_labels = [[self.tag_to_idx[tag] if tag in self.tag_to_idx else self.tag_to_idx['<unk>'] for tag in sentence] for sentence in self.train_labels]
        self.dev_labels = [[self.tag_to_idx[tag] if tag in self.tag_to_idx else self.tag_to_idx['<unk>'] for tag in sentence] for sentence in self.dev_labels]
        self.test_labels = [[self.tag_to_idx[tag] if tag in self.tag_to_idx else self.tag_to_idx['<unk>'] for tag in sentence] for sentence in self.test_labels]

    def preprocess_sentence(self, sentence):
        tokens = [self.word_to_idx['<sos>']]
        for word in sentence.split():
            tokens.append(self.word_to_idx[word] if word in self.word_to_idx else self.word_to_idx['<unk>'])
        tokens.append(self.word_to_idx['<eos>'])
        return tokens
    