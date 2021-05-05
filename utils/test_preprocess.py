#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-9-21 ??10:18
# adapted by: scantor96
import puz
import pickle
import numpy as np
import json
from utils.util import read_data,Vocabulary
from pre_hypm import read_hypernyms,get_hnym

def make_shortlist(puzfile):
    puzzle = puz.read(puzfile)
    answer_list = []
    num = puzzle.clue_numbering()
    for clue in num.across:
        answer = "".join(puzzle.solution[clue["cell"] + i] for i in range(clue["len"]))
        answer_list.append(answer.lower())
    for clue in num.down:
        answer = "".join(puzzle.solution[clue["cell"] + i * num.width] for i in range(clue["len"]))
        answer_list.append(answer.lower())
    with open("/home/cantors2/Documents/xwordPytorch/data/shortlist_test.txt","w") as out_file:
        for row in answer_list:
            out_file.write(row)
            out_file.write("\n")

def make_txt(puzfile):
    puzzle = puz.read(puzfile)
    answer_list = []
    num = puzzle.clue_numbering()
    for clue in num.across:
        answer = "".join(puzzle.solution[clue["cell"] + i] for i in range(clue["len"]))
        answer_list.append([answer, clue['clue']])  
    for clue in num.down:
        answer = "".join(puzzle.solution[clue["cell"] + i * num.width] for i in range(clue["len"]))
        answer_list.append([answer, clue['clue']])
    with open("/home/cantors2/Documents/xwordPytorch/data/test.txt","w+") as out_file:
        for row in answer_list:
            out_file.write(row[0].lower())
            out_file.write("\t")
            out_file.write(row[1])
            out_file.write("\n")
    out_file.close()
    
def make_pkl():
    test_defs = "/home/cantors2/Documents/xwordPytorch/data/test.txt"
    test_save = "/home/cantors2/Documents/xwordPytorch/data/processed/test.pkl"
    embedding = "/home/cantors2/Documents/xwordPytorch/data/processed/embedding.pkl"
    vectors = []
    data = read_data(test_defs)
    vocab = Vocabulary()
    vocab.load("/home/cantors2/Documents/xwordPytorch/data/processed/vocab.json")
    with open(embedding, 'rb') as infile:
        word_embedding = pickle.load(infile)
    for element in data:
        vectors.append(word_embedding[vocab.encode(element[0])])
    with open(test_save, 'wb') as outfile:
        pickle.dump(np.array(vectors), outfile)
        outfile.close()
        
def make_hyp_files():
    hypernym_data = read_hypernyms("/home/cantors2/Documents/xwordPytorch/data/bag_of_hypernyms.txt")
    vocab = Vocabulary()
    vocab.load("/home/cantors2/Documents/xwordPytorch/data/processed/vocab.json")
    word2hym, hym_weights = get_hnym(hypernym_data, vocab)
    defs = "/home/cantors2/Documents/xwordPytorch/data/test.txt"
    top_k = 5
    save = "/home/cantors2/Documents/xwordPytorch/data/processed/test_hyp.json"
    save_hypm = "/home/cantors2/Documents/xwordPytorch/data/processed/test_word2hym.json"
    save_weights = "/home/cantors2/Documents/xwordPytorch/data/processed/test_hym_weights.json"
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
    
