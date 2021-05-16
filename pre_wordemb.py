#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-11-14 ??7:14
# adapted by: scantor96
import time
import pickle
import numpy as np
from tqdm import tqdm
from gensim.models.keyedvectors import KeyedVectors
from utils.datasets import Vocabulary
from utils.util import get_time_dif
import os

argsvocab = ".../data/processed/vocab.json"
w2v = os.path.join(".../data/word2vec/GoogleNews-vectors-negative300.bin.gz")
emb_dim = 300
start_time = time.time()
save = ".../data/processed/embedding.pkl"
print('Start prepare word embeddings at {}'.format(time.asctime(time.localtime(start_time))))
vocab = Vocabulary()
vocab.load(argsvocab)
word2vec = KeyedVectors.load_word2vec_format(w2v, binary=True)
init_embedding = np.random.uniform(-1.0, 1.0, (len(vocab), emb_dim))
for word in tqdm(vocab.token2id.keys()):
    if word in word2vec:
        init_embedding[vocab.encode(word)] = word2vec[word]
init_embedding[vocab.encode('<pad>')] = np.zeros([emb_dim])
with open(save, 'wb') as f:
    pickle.dump(init_embedding, f)
    f.close()
time_dif = get_time_dif(start_time)
print("Finished!Prepare word embeddings time usage:", time_dif)
