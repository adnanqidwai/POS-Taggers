import os
import sys
import torch
from FF import PreprocessorFFN
from LSTM import PreprocessorLSTM
import json

if len(sys.argv) != 2:
    print("[Invalid number of arguments] choose between -f or -r")
    exit(1)

arg = sys.argv[1][1:]

model = None
model_name = None

if arg == "f":
    model_name = "./models/FFNModel.pt"
elif arg == "r":
    model_name = "./models/LSTMModel.pt"
else:
    print("[Invalid model type] choose between -f or -r")
    exit(1)

if os.path.exists(model_name):
    model = torch.load(model_name)
else:
    print("Model does not exist")
    exit(1)

idx_to_tag = json.load(open('./vocab/idx_to_tag.json', 'r'))
idx_to_word = json.load(open('./vocab/idx_to_word.json', 'r'))
word_to_idx = json.load(open('./vocab/word_to_idx.json', 'r'))
tag_to_idx = json.load(open('./vocab/tag_to_idx.json', 'r'))

def preprocess_sentence(sentence, p = 1, s = 1):
    tokens = [word_to_idx['<sos>']]*p
    for word in sentence.split():
        tokens.append(int(word_to_idx[word]) if word in word_to_idx else int(word_to_idx['<unk>']))
    tokens.extend([word_to_idx['<eos>']]*(s))
    return tokens

model.eval()

p = 1
s = 1


if (arg == "f"):
    p = 2
    s = 2

context_window = p + 1 + s

sentence = input(">  ")
tokens = preprocess_sentence(sentence, p, s)
print()
sentence = sentence.split()

tokens = torch.tensor(tokens).unsqueeze(0)

output = None

if arg == "r":
    with torch.no_grad():
        output = model(tokens).argmax(dim=2)
        output = list(output[0][1:])

else :
    with torch.no_grad():
        output = []
        maxLen = tokens.shape[1]
        for i in range(maxLen - context_window + 1):
            context = tokens[:, i:i + context_window]
            pred = model(context)
            output.append(pred.argmax(dim=1))

for i in range(len(sentence)):
    print(f"{sentence[i]}: {idx_to_tag[str(output[i].item())]}")
print()