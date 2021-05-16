import torch
from torch import nn
from model.layers import Hypernym, Hidden, Gated
from regularize.embed_dropout import embedded_dropout
from regularize.locked_dropout import LockedDropout
from regularize.weight_dropout import WeightDrop
import numpy as np
import pickle


class RNNModel(nn.Module):
    def __init__(self, rnn_type,nlayers,hidim,emdim,use_input,use_hidden, \
                 use_gated,dropout,dropouth,dropouti, \
                 dropoute,vocab_size,fix_embeddings, wdrop,tied, \
                 batch_size,use_ch,use_he,w2v_weights):
        super(RNNModel, self).__init__()

        #self.params = params
        self.rnn_type = rnn_type
        self.n_layers = nlayers
        self.hi_dim = hidim
        self.embedding_dim = emdim
        self.use_input = use_input
        self.use_hidden = use_hidden
        self.use_gated = use_gated
        self.vocab_size = vocab_size
        self.fix_embeddings = fix_embeddings
        self.wdrop = wdrop
        self.tied = tied
        self.batch_size = batch_size
        self.use_ch = use_ch
        self.use_he = use_he
        self.is_conditioned = self.use_input
        self.is_conditioned += self.use_hidden
        self.is_conditioned += self.use_gated
        self.device = torch.device('cpu')

        assert self.use_input + self.use_hidden + \
               self.use_gated <= 1, "Too many conditionings used"

        self.drop = nn.Dropout(dropout)
        self.lockdrop = LockedDropout()
        self.hdrop = nn.Dropout(dropouth)
        self.idrop = nn.Dropout(dropouti)
        self.edrop = nn.Dropout(dropoute)
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.w2v_weights = w2v_weights
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        if self.w2v_weights:
           with open(self.w2v_weights, 'rb') as infile:
                pretrain_emb = pickle.load(infile)
                infile.close()
           self.embedding = self.embedding.from_pretrained(
                torch.from_numpy(pretrain_emb).float()
            )
        else:
            self.embedding.weight.data.copy_(
            torch.from_numpy(
                self.random_embedding(self.vocab_size, self.embedding_dim)
            )
        )
        self.embedding.weight.requires_grad = not self.fix_embeddings

        self.ch_dim = 0
        self.he_dim = 0
        # ch
        
#        if self.use_ch:
#            n_ch_tokens=kwargs["n_ch_tokens"],
#            ch_maxlen=kwargs["ch_maxlen"],
#            ch_emb_size=kwargs["ch_emb_size"],
#            ch_feature_maps=kwargs["ch_feature_maps"],
#            ch_kernel_sizes=kwargs["ch_kernel_sizes"]
#            self.ch = CharCNN(n_ch_tokens,ch_maxlen,ch_emb_size,ch_feature_maps,ch_kernel_sizes,dropout).to(self.device)
#            self.ch_dim = sum(self.ch_feature_maps)
        
        
        # he
        if self.use_he:
            self.he_dim = self.embedding_dim
            self.he = Hypernym(self.embedding_dim, self.embedding, self.device)
        
        concat_embedding_dim = self.embedding_dim + self.ch_dim + self.he_dim
        self.word2hidden = nn.Linear(concat_embedding_dim, self.hi_dim)
        
        if self.use_input:
            self.embedding_dim = self.embedding_dim + concat_embedding_dim
        if self.use_hidden:
            self.hidden = Hidden(
                in_size=concat_embedding_dim + self.hi_dim,
                out_size=self.hi_dim
            )
        if self.use_gated:
            self.gated = Gated(
                cond_size=concat_embedding_dim,
                hidden_size=self.hi_dim
            )
        if self.rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, self.rnn_type)(self.embedding_dim, self.hi_dim, self.n_layers, dropout=0)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                               options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(self.embedding_dim, self.hi_dim, self.n_layers, nonlinearity=nonlinearity, dropout=0)
        if self.wdrop != 0:
            self.rnn = WeightDrop(self.rnn, ['weight_hh_l0'], dropout=self.wdrop)
        self.decoder = nn.Linear(self.hi_dim, self.vocab_size)
        if self.tied:
            if self.hi_dim != self.embedding_dim:
                print(self.hi_dim)
                print(self.embedding_dim)
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.embedding.weight
        self.init_weights()

    def forward(self, inputs, init_hidden, return_h=False):
        word = inputs['word']
        seq = inputs['seq']
        if self.use_input:
            input_vectors = inputs["input_vectors"]
        if self.use_ch:
            chars = inputs['chars']
        if self.use_he:
            hynm = inputs['hypm']
            hynm_weights = inputs['hypm_weights']
        batch_size = word.size(0)

        word_emb = embedded_dropout(self.embedding, word, dropout=self.dropoute if self.training else 0)
        seq_emb = embedded_dropout(self.embedding, seq, dropout=self.dropoute if self.training else 0)
        seq_emb = self.lockdrop(seq_emb, self.dropouti)
        if self.use_ch:
            char_embeddings = self.ch(chars)
            word_emb = torch.cat(
                [word_emb, char_embeddings], dim=-1)
        if self.use_he:
            hynm_embeddings = self.he([hynm, hynm_weights])
            word_emb = torch.cat(
                [word_emb, hynm_embeddings], dim=-1)
        if init_hidden is not None:
            hidden = init_hidden
        else:
            hidden = self.init_hidden(word_emb, batch_size, self.n_layers, self.hi_dim)
        raw_outputs = []
        lock_outputs = []
        outputs = []
        for time_step in range(seq.size(0)):
            if time_step != 0:
                raw_outputs = []
                lock_outputs = []
            inp_seq = seq_emb[time_step, :, :].view(1, batch_size, -1)
            if self.use_input:
                inp_seq = torch.cat([torch.unsqueeze(word_emb, 0), inp_seq], dim=-1)
                outs, hidden = self.rnn(inp_seq, hidden)
                raw_outputs.append(outs)
                outs = self.lockdrop(outs, self.dropout)
                lock_outputs.append(outs)
            else:
                outs, hidden = self.rnn(inp_seq, hidden)
                raw_outputs.append(outs)
                outs = self.lockdrop(outs, self.dropout)
                lock_outputs.append(outs)
                if self.use_hidden:
                    hidden = self.hidden(self.rnn_type, self.n_layers, word_emb, hidden)
                if self.use_gated:
                    hidden = self.gated(self.rnn_type, self.n_layers, word_emb, hidden)
            if time_step == 0:
                rnn_hs = raw_outputs
                dropped_rnn_hs = lock_outputs
            else:
                for i in range(len(rnn_hs)):
                    rnn_hs[i] = torch.cat((rnn_hs[i], raw_outputs[i]), 0)
                    dropped_rnn_hs[i] = torch.cat((dropped_rnn_hs[i], lock_outputs[i]), 0)
            outputs.append(outs)
        outputs = torch.cat(outputs, dim=0)
        outputs = outputs.view(outputs.size(0) * outputs.size(1), outputs.size(2))
        decoded = self.decoder(self.drop(outputs))
        if return_h:
            return decoded, hidden, rnn_hs, dropped_rnn_hs
        return decoded, hidden

    def init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.constant_(self.word2hidden.bias, 0.0)
        nn.init.xavier_normal_(self.word2hidden.weight)
        # maybe not use init decoder?
        nn.init.constant_(self.decoder.bias, 0.0)
        nn.init.xavier_normal_(self.decoder.weight)
        if self.use_hidden:
            self.hidden.init_hidden()
        if self.use_gated:
            self.gated.init_gated()
#        if self.use_ch:
#            self.ch.init_ch()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for i in range(vocab_size):
            pretrain_emb[i, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def init_hidden(self, v, batch_size, num_layers, hidden_dim):
        hidden = self.word2hidden(v).view(-1, batch_size, hidden_dim)
        hidden = hidden.expand(num_layers, batch_size, hidden_dim).contiguous()
        if self.rnn_type == 'LSTM':
            ###############################################h,c fan
            h_c = hidden
            h_h = torch.zeros_like(h_c)
            hidden = (h_h, h_c)
        return hidden
