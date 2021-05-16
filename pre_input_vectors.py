import time
import pickle
import numpy as np
from utils.datasets import Vocabulary
from utils.util import read_data


embedding = ".../data/processed/embedding.pkl"
train_defs = ".../data/defs/train.txt"
test_defs = ".../data/defs/test.txt"
valid_defs = ".../data/defs/valid.txt"

train_save = ".../data/processed/train.pkl"
valid_save = ".../data/processed/valid.pkl"
test_save = ".../data/processed/test.pkl"
start_time = time.time()
print('Start prepare input vectors at {}'.format(time.asctime(time.localtime(start_time))))

vectors = []
data = read_data(train_defs)
vocab = Vocabulary()
vocab.load(".../data/processed/vocab.json")
with open(embedding, 'rb') as infile:
    word_embedding = pickle.load(infile)
for element in data:
    vectors.append(word_embedding[vocab.encode(element[0])])
with open(train_save, 'wb') as outfile:
    pickle.dump(np.array(vectors), outfile)
    outfile.close()
