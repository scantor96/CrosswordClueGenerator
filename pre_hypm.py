#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-9-13 ??1:35
# adapted by: scantor96
import numpy as np
import time
import json
import pickle
from collections import defaultdict
from utils.datasets import Vocabulary
from utils.util import get_time_dif, read_data
    
def read_hypernyms(file_path):
    """Read hypernyms"""
    hnym_data = defaultdict(list)
    with open(file_path, 'r') as fr:
        for line in fr:
            line = line.strip().split('\t')
            word = line[0]
            line = line[1:]
            assert len(line) % 2 == 0
            for i in range(int(len(line) / 2)):
                hnym = line[2 * i]
                weight = line[2 * i + 1]
                hnym_data[word].append((hnym, weight))
    return hnym_data


def get_hnym(hnym_data, vocab):
    """Get hypernyms and weights"""
    word2hnym = defaultdict(list)
    hnym_weights = defaultdict(list)
    for key, value in hnym_data.items():
        weight_sum = sum([float(w) for h, w in value])
        for hnym, weight in value:
            word2hnym[key].append(vocab.encode(hnym))
            hnym_weights[key].append(float(weight) / weight_sum)
    return word2hnym, hnym_weights


start_time = time.time()
print('Start prepare word hypernyms and weights at {}'.format(time.asctime(time.localtime(start_time))))
hypernym_data = read_hypernyms(".../data/bag_of_hypernyms.txt")
vocab = Vocabulary()
vocab.load(".../data/processed/vocab.json")
word2hym, hym_weights = get_hnym(hypernym_data, vocab)
defs = ".../data/defs/train.txt"
top_k = 5
save = ".../data/processed/train_hyp.json"
save_hypm = ".../data/processed/train_word2hym.json"
save_weights = ".../data/processed/train_hym_weights.json"
#for i in defs:
data = read_data(defs)
hnym = np.zeros((len(data), top_k))
hnym_weights = np.zeros_like(hnym)
assert len(data) == len(hnym)
for l, (word, _) in enumerate(data):
    for j, h in enumerate(word2hym[word][:top_k]):
        hnym[l][j] = h
    for k, weight in enumerate(hym_weights[word][:top_k]):
        hnym_weights[l][k] = weight
with open(save, 'wb') as outfile:
    pickle.dump([hnym, hnym_weights], outfile)
    outfile.close()
with open(save_hypm, 'w') as f:
    json.dump(word2hym, f, indent=4)
    f.close()
with open(save_weights, 'w') as f:
    json.dump(hym_weights, f, indent=4)
    f.close()
time_dif = get_time_dif(start_time)
print("Finished!Prepare word hypernyms time usage:", time_dif)
