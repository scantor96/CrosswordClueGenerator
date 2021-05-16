import time
import codecs
from collections import defaultdict
from datetime import timedelta
import os
from utils import constants
import json

def pad(seq, size, value):
    if len(seq) < size:
        seq.extend([value] * (size - len(seq)))
    return seq

class Vocabulary:
    """Word/char vocabulary"""

    def __init__(self):
        self.token2id = {
            constants.PAD: constants.PAD_IDX,
            constants.UNK: constants.UNK_IDX,
            constants.BOS: constants.BOS_IDX,
            constants.EOS: constants.EOS_IDX,
        }
        self.id2token = {
            constants.PAD_IDX: constants.PAD,
            constants.UNK_IDX: constants.UNK,
            constants.BOS_IDX: constants.BOS,
            constants.EOS_IDX: constants.EOS,
        }
        self.token_maxlen = -float("inf")

    def encode(self, tok):
        #if tok in self.token2id:
        return self.token2id[tok]
        #else:
        #    return constants.UNK_IDX

    def decode(self, idx):
        if id in self.id2token:
            return self.id2token[id]
        else:
            raise ValueError("No such idx: {0}".format(idx))

    def encode_seq(self, seq):
        e_seq = []
        for s in seq:
            e_seq.append(self.encode(s))
        return e_seq

    def decode_seq(self, seq):
        d_seq = []
        for i in seq:
            d_seq.append(self.decode(i))
        return d_seq

    def add_token(self, tok):
        if tok not in self.token2id:
            self.token2id[tok] = len(self.token2id)
            self.id2token[len(self.id2token)] = tok

    def save(self, path):
        with open(path, "w") as outfile:
            json.dump([self.id2token, self.token_maxlen], outfile, indent=4)
        outfile.close()

    def load(self, path):
        with open(path, 'r') as infile:
            self.id2token, self.token_maxlen = json.load(infile)
            self.id2token = {int(k): v for k, v in self.id2token.items()}
            self.token2id = {}
            for i in self.id2token.keys():
                self.token2id[self.id2token[i]] = i

    def __len__(self):
        return len(self.token2id)

def get_time_dif(start_time):
    """Compute time"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def read_data(file_path):
    """Read definitions file"""
    content = []
    print(file_path)
    pre = "/home/cantors2/Documents/xwordPytorch/data/defs"
    with codecs.open(os.path.join(pre,file_path), 'r', 'utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            defs=[]
            definition=line[-1].split(" ")
            for d in definition:
                defs.append(d)
            content.append([line[0], defs])
    return content


def read_hypernyms(file_path):
    """Read hypernyms file"""
    hyp_token = []
    hnym_data = defaultdict(list)
    with open(file_path, 'r') as fr:
        for line in fr:
            line = line.strip().split('\t')
            word = line[0]
            hyp_token.append(word)
            line = line[1:]
            assert len(line) % 2 == 0
            for i in range(int(len(line) / 2)):
                hnym = line[2 * i]
                hyp_token.append(hnym)
                weight = line[2 * i + 1]
                hnym_data[word].append([hnym, weight])
    return hnym_data, hyp_token

if __name__=="__main__":
    a=read_hypernyms("../data/bag_of_hypernyms.txt")
    print()
    
