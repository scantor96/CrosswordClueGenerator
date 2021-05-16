#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-9-13 ??1:35
# adapted by: scantor96
import os
import numpy as np
import torch
import time
from torch import nn
import json
from tqdm import tqdm
from model.model import RNNModel
from utils.datasets import DefinitionModelingDataset, DefinitionModelingCollate
from utils import constants
from utils.util import get_time_dif
from torch.utils.data import DataLoader

# Read all arguments and prepare all stuff for training
params={}

# Common data arguments
# location of vocabulary file
voc = ".../data/processed/vocab.json"
params['voc']=voc

# Definitions data arguments
# location of txt file with train definitions.
train_defs = ".../data/defs/train.txt"
params['train_defs']=train_defs

# location of txt file with metrics definitions.
eval_defs = ".../data/defs/valid.txt"
params['eval_defs']=eval_defs    

# location of txt file with test definitions
test_defs = ".../data/defs/test.txt"  
params['test_defs']=test_defs 
   
# location of train vectors for Input conditioning
input_train = ".../data/processed/train.pkl"
params['input_train']=input_train
    
# location of metrics vectors for Input conditioning
input_eval = ".../data/processed/valid.pkl"
params['input_eval']=input_eval    

# location of test vectors for Input conditioning
input_test = ".../data/processed/test.pkl"  
params['input_test']=input_test    

# location of train hypernyms for Hypernyms conditioning
hypm_train=".../data/processed/train_hyp.json" 
params['hypm_train']=hypm_train 

# location of metrics hypernyms for Hypernyms conditioning
hypm_eval=".../data/processed/valid_hyp.json"   
params['hypm_eval']=hypm_eval  

# location of test hypernyms for Hypernyms conditioning
hypm_test=".../data/processed/test_hyp.json"  
params['hypm_test']=hypm_test   

# location of CH vocabulary file
ch_voc=".../data/processed/char_vocab.json"
params['ch_voc']=ch_voc

# Model parameters arguments
# type of recurrent neural network(LSTM,GRU)'
rnn_type='LSTM'
params['rnn_type']=rnn_type

# size of word embeddings
emdim=300
params['emdim']=emdim

# numbers of hidden units per layer
hidim=300
params['hidim']=hidim

# number of recurrent neural network layers
nlayers=2
params['nlayers']=nlayers

# whether to use Seed conditioning or not
use_seed=False
params['use_seed']=use_seed

# whether to use Input conditioning or not
use_input=False
params['use_input']=use_input

# whether to use Hidden conditioning or not
use_hidden=True
params['use_hidden']=use_hidden

# whether to use Gated conditioning or not
use_gated = False
params['use_gated']=use_gated

# use character level CNN
use_ch = False
params['use_ch']=use_ch

# use hypernym embeddings
use_he=True
params[use_he]=use_he

# Training and dropout arguments
# initial learning rate
lr=0.001
params['lr']=lr

# factor to decay lr
decay_factor=0.1
params['decay_factor']=decay_factor

# after number of patience epochs - decay lr
decay_patience=0
params['decay_patience']=decay_patience

# value to clip norm of gradients to
clip=5
params['clip']=clip

# upper epoch limit
epochs=50
params['epochs']=epochs

# batch size
batch_size=64
params['batch_size']=batch_size

# tie the word embedding and softmax weights
tied=True
params['tied']=tied

# random seed
random_seed=22222
params['random_seed']=random_seed

# dropout applied to layers (0 = no dropout)
dropout=0.3
params['dropout']=dropout

# dropout for rnn layers (0 = no dropout)
dropouth=0.1
params['dropouth']=dropouth

# dropout for input embedding layers (0 = no dropout)
dropouti=0.1
params['dropouti']=dropouti

# dropout to remove words from embedding layer (0 = no dropout)
dropoute=0.1
params['dropoute']=dropoute

# amount of weight dropout to apply to the RNN hidden to hidden matrix
wdrop=0.2
params['wdrop']=wdrop

# weight decay applied to all weights
wdecay=1.2e-6
params['wdecay']=1.2e-6

# alpha L2 regularization on RNN activation (alpha = 0 means no regularization)
alpha=0
params['alpha']=alpha

# beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)
beta=1
params['beta']=beta

# where to save all stuff about training
exp_dir=".../checkpoints/"

# path to pretrained embeddings to init
w2v_weights=".../data/processed/embedding.pkl"

# whether to update embedding matrix or not
fix_embeddings=True
params['fix_embeddings']=fix_embeddings

# use CUDA
cuda=False

train_dataset = DefinitionModelingDataset(
    file=train_defs,
    vocab_path=voc,
    input_vectors_path=input_train,
    ch_vocab_path=ch_voc,
    use_seed=use_seed,
    hypm_path=hypm_train,
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=DefinitionModelingCollate,
    shuffle=True,
    num_workers=2
)
valid_dataset = DefinitionModelingDataset(
    file=eval_defs,
    vocab_path=voc,
    input_vectors_path=input_eval,
    ch_vocab_path=ch_voc,
    use_seed=use_seed,
    hypm_path=hypm_eval,
)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    collate_fn=DefinitionModelingCollate,
    shuffle=True,
    num_workers=2
)

if use_input or use_hidden or use_gated:
    assert input_train is not None, ("--use_input or "
                                             "--use_hidden or "
                                             "--use_gated is used "
                                             "--input_train is required")
    assert input_eval is not None, ("--use_input or "
                                            "--use_hidden or "
                                            "--use_gated is used "
                                            "--input_eval is required")
    assert input_test is not None, ("--use_input or "
                                            "--use_hidden or "
                                            "--use_gated is used "
                                            "--input_test is required")
    input_dim = train_dataset.input_vectors.shape[1]

if use_ch:
    assert ch_voc is not None, ("--ch_voc is required "
                                        "if --use_ch")
    assert ch_emb_size is not None, ("--ch_emb_size is required "
                                             "if --use_ch")
    assert ch_feature_maps is not None, ("--ch_feature_maps is "
                                                 "required if --use_ch")
    assert ch_kernel_sizes is not None, ("--ch_kernel_sizes is "
                                                 "required if --use_ch")

    n_ch_tokens = len(train_dataset.ch_voc.token2id)
    ch_maxlen = train_dataset.ch_voc.token_maxlen + 2
if use_he:
    assert hypm_train is not None, ("--use_he is used "
                                            "--hypm_train is required")
    assert hypm_eval is not None, ("--use_he is used "
                                           "--hypm_eval is required")
    assert hypm_test is not None, ("--use_he is used "
                                           "--hypm_test is required")

vocab_size = len(train_dataset.voc.token2id)
# Set the random seed manually for reproducibility
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    if not cuda:
        print('WARNING:You have a CUDA device,so you should probably run with --cuda')
    else:
        torch.cuda.manual_seed(random_seed)
device = torch.device('cuda' if cuda else 'cpu')


def train():
    print('=========model architecture==========')
    model = RNNModel( \
        rnn_type, nlayers, hidim, emdim, use_input, use_hidden, \
        use_gated, dropout, dropouth, dropouti, \
        dropoute, vocab_size, fix_embeddings, wdrop,tied, batch_size,\
        use_ch,use_he,w2v_weights
        ).to(device)
    print(model)
    print('=============== end =================')
    loss_fn = nn.CrossEntropyLoss(ignore_index=constants.PAD_IDX)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr, weight_decay=wdecay)
    print('Training and evaluating...')
    start_time = time.time()
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    best_ppl = 9999999
    last_improved = 0
    require_improvement = 5
    with open(exp_dir + "params.json", "w") as outfile:
        json.dump(params, outfile, indent=4)
    for epoch in range(epochs):
        model.training = True
        loss_epoch = []
        for batch, inp in enumerate(tqdm(train_dataloader, desc='Epoch: %03d' % (epoch + 1), leave=False)):
            data = {
                'word': torch.from_numpy(inp['word']).long().to(device),
                'seq': torch.t(torch.from_numpy(inp['seq'])).long().to(device)
            }
            if model.use_input:
                data["input_vectors"] = torch.from_numpy(inp['input']).to(device)
            if model.use_ch:
                data["chars"] = torch.from_numpy(inp['chars']).long().to(device)
            if model.use_he:
                data["hypm"] = torch.from_numpy(inp['hypm']).long().to(device)
                data["hypm_weights"] = torch.from_numpy(inp['hypm_weights']).float().to(device)
            targets = torch.t(torch.from_numpy(inp['target'])).to(device)
            output, hidden, rnn_hs, dropped_rnn_hs = model(data, None, return_h=True)
            loss = loss_fn(output, targets.reshape(2))
            optimizer.zero_grad()
            #Activiation Regularization
            if alpha:
                loss = loss + sum(
                     alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            if beta:
                loss = loss + sum(beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            loss.backward()
            # `clip_grad_norm`
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            loss_epoch.append(loss.item())
        train_loss = np.mean(loss_epoch)
        train_ppl = np.exp(train_loss)
        valid_loss, valid_ppl = evaluate(model, valid_dataloader, device)
        
        if valid_ppl < best_ppl:
            best_ppl = valid_ppl
            last_improved = epoch
            torch.save(model, exp_dir +
                       'defseq_model_params_%s_min_ppl.pkl' % (epoch + 1)
                       )
            improved_str = '*'
        else:
            improved_str = ''
        time_dif = get_time_dif(start_time)
        msg = 'Epoch: {0:>6},Train Loss: {1:>6.6}, Train Ppl: {2:>6.6},' \
              + ' Val loss: {3:>6.6}, Val Ppl: {4:>6.6},Time:{5} {6}'
        print(msg.format(epoch + 1, train_loss, train_ppl, valid_loss, valid_ppl, time_dif, improved_str))
        if epoch - last_improved > require_improvement:
            print("No optimization for a long time, auto-stopping...")
            break
    return 1


def evaluate(model, dataloader, device='cpu'):
    model.training = False
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        total_loss = []
        for inp in dataloader:
            data = {
                'word': torch.tensor(inp['word']).to(device),
                'seq': torch.t(torch.tensor(inp['seq'])).to(device),
            }
            if model.use_input:
                data["input_vectors"] = torch.from_numpy(inp['input']).to(device)
            if model.use_ch:
                data["chars"] = torch.from_numpy(inp['chars']).long().to(device)
            if model.use_he:
                data["hypm"] = torch.from_numpy(inp['hypm']).long().to(device)
                data["hypm_weights"] = torch.from_numpy(inp['hypm_weights']).float().to(device)
            targets = torch.t(torch.from_numpy(inp['target'])).to(device)
            output, hidden = model(data, None)
            loss = loss_fn(output, targets.reshape(-1))
            total_loss.append(loss.item())
    return np.mean(total_loss), np.exp(np.mean(total_loss))


if __name__ == "__main__":
    train()
