import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, context_size, emb_size = 128, dropout = 0.5, extra_hidden = False):
        super(FFN, self).__init__()
        self.extra_hidden = extra_hidden
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.fc1 = nn.Linear(context_size*emb_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.embedding(x).reshape(x.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        if self.extra_hidden:
            out = F.relu(self.fc3(out))
            out = self.dropout(out)
        out = self.fc2(out)
        return out


class PreprocessorFFN():
    def __init__(self, train_set, val_set, test_set,p = 2,s = 3):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.p = p
        self.s = s

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
            self.train_sentences.append(['<sos>']*(self.p) + [self._word_cleaner(word['form'].lower()) for word in sentence] + ['<eos>']*(self.s))
            self.train_labels.append([word['upostag'] for word in sentence])

        for sentence in self.val_set:
            self.dev_sentences.append(['<sos>']*(self.p) + [self._word_cleaner(word['form'].lower()) for word in sentence] + ['<eos>']*(self.s))
            self.dev_labels.append([word['upostag'] for word in sentence])

        for sentence in self.test_set:
            self.test_sentences.append(['<sos>']*(self.p) + [self._word_cleaner(word['form'].lower()) for word in sentence] + ['<eos>']*(self.s))
            self.test_labels.append([word['upostag'] for word in sentence])

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
        tokens = [self.word_to_idx['<sos>']]*self.p
        for word in sentence.split():
            tokens.append(self.word_to_idx[word] if word in self.word_to_idx else self.word_to_idx['<unk>'])
        tokens.extend([self.word_to_idx['<eos>']]*(self.s))
        return tokens