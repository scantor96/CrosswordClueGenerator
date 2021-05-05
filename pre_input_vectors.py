#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-11-20 ??9:37
# adapted by: scantor96
import time
import pickle
import numpy as np
from utils.datasets import Vocabulary
from utils.util import read_data


embedding = "/home/cantors2/Documents/xwordPytorch/data/processed/embedding.pkl"
train_defs = ("/home/cantors2/Documents/xwordPytorch/data/defs/train.txt")
test_defs = ("/home/cantors2/Documents/xwordPytorch/data/defs/test.txt")
valid_defs = ("/home/cantors2/Documents/xwordPytorch/data/defs/valid.txt")

train_save = "/home/cantors2/Documents/xwordPytorch/data/processed/train.pkl"
valid_save = "/home/cantors2/Documents/xwordPytorch/data/processed/valid.pkl"
test_save = "/home/cantors2/Documents/xwordPytorch/data/processed/test.pkl"
start_time = time.time()
print('Start prepare input vectors at {}'.format(time.asctime(time.localtime(start_time))))

vectors = []
data = read_data(train_defs)
vocab = Vocabulary()
vocab.load("/home/cantors2/Documents/xwordPytorch/data/processed/vocab.json")
with open(embedding, 'rb') as infile:
    word_embedding = pickle.load(infile)
for element in data:
    vectors.append(word_embedding[vocab.encode(element[0])])
with open(train_save, 'wb') as outfile:
    pickle.dump(np.array(vectors), outfile)
    outfile.close()
