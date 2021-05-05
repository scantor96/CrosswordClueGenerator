#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-9-21 ??10:18
# adapted by: scantor96
import torch
import os
import json
from tqdm import tqdm
from model.model import RNNModel
from torch.utils.data import DataLoader
from utils.datasets import DefinitionModelingDataset, DefinitionModelingCollate
from utils.datasets import Vocabulary
from utils.test_preprocess import make_shortlist,make_txt,make_pkl,make_hyp_files
import time

# path to saved model params
params = "/home/cantors2/Documents/xwordPytorch/checkpoints/params.json"

# path to saved model weights
ckpt = "/home/cantors2/Documents/xwordPytorch/checkpoints/defseq_model_params_min_ppl.pkl"

# temperature to use in sampling
tau = 1

# path to binary w2v file
w2v_binary_path = "/home/cantors2/Documents/xwordPytorch/data/processed/embedding.pkl"

# where to save generate file
gen_dir = "/home/cantors2/Documents/xwordPytorch/gen/"

# generate file name
gen_name="gen1.txt"

# load puz + puz json
input_puz = input("Enter puzzle path: ")
start_time = time.time()
test = make_txt(input_puz)
make_shortlist(input_puz)
generate_list = "/home/cantors2/Documents/xwordPytorch/data/shortlist_test.txt"
make_pkl()
make_hyp_files()

with open(params, "r") as infile:
    model_params = json.load(infile)

dataset = DefinitionModelingDataset(
    file=generate_list,
    vocab_path="/home/cantors2/Documents/xwordPytorch/data/processed/vocab.json",
    input_vectors_path="/home/cantors2/Documents/xwordPytorch/data/processed/test.pkl",
    ch_vocab_path="/home/cantors2/Documents/xwordPytorch/data/processed/char_vocab.json",
    use_seed=model_params["use_seed"],
    hypm_path="/home/cantors2/Documents/xwordPytorch/data/processed/test_hyp.json",
    mode="gen"
)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    collate_fn=DefinitionModelingCollate,
    num_workers=2
)
device = torch.device("cpu")
model = RNNModel(\
        "GRU",2,300,300,True,False,False,0.3,0.1,0.1,0.1,len(dataset),True, \
            0.1,False,10,False,True,w2v_binary_path)
model.to(device)
model = torch.load(ckpt)
voc = Vocabulary()
voc.load("/home/cantors2/Documents/xwordPytorch/data/processed/vocab.json")


def generate(model, dataloader, idx2word, strategy='greedy',max_len=20):
    model.training = False
    for inp in tqdm(dataloader, desc='Generate Definitions', leave=False):
        word_list = []
        data = {
            'word': torch.from_numpy(inp['word']).to(device),
            'seq': torch.t(torch.from_numpy(inp["seq"])).to(device),
        }
        if model.use_input:
            data["input_vectors"] = torch.from_numpy(inp['input']).to(device)
        if model.use_ch:
            data["chars"] = torch.from_numpy(inp['chars']).long().to(device)
        if model.use_he:
            data["hypm"] = torch.from_numpy(inp['hypm']).long().to(device)
            data["hypm_weights"] = torch.from_numpy(inp['hypm_weights']).float().to(device)

        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)
        def_word = [idx2word[inp['word'][0]], "\t"]
        word_list.extend(def_word)
        hidden = None
        for i in range(max_len):
            output, hidden = model(data, hidden)
            word_weights = output.squeeze().div(tau).exp().cpu()
            if strategy == 'greedy':
                word_idx = torch.argmax(word_weights,dim=-1,keepdim=False)
            elif strategy == 'multinomial':
                word_idx = torch.multinomial(word_weights,10,replacement=True)[0]
            if word_idx == 3:
                break
            else:
                data['seq'].fill_(word_idx)
                word = idx2word[word_idx.item()]
                word_list.append(word)
        with open(gen_dir + gen_name, "a") as f:
            for item in word_list:
                f.write(item + " ")
            f.write("\t")
            f.write("\n")
            f.close()
    print("Finished!")
    return 1


if __name__ == "__main__":
    print('=========model architecture==========')
    print(model)
    print('=============== end =================')
    generate(model, dataloader, voc.id2token)
