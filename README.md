# CrosswordClueGenerator

# Preprocessing
1. Download Google News Vectors https://code.google.com/archive/p/word2vec/ and save to .../data/word2vec/
For next steps, input appropriate absolute paths wherever a ".../xxxx/..." spot is.
3. Run pre_vocab.py
4. Run pre_wordemb.py
5. Run pre_hypm.py, changing to train/valid/text files each for the defs, save, save_hypm, save_weights variables 
6. Run pre_input_vectors.py

# Training
Run train.py with the following parameters:
- rrn_type = "LSTM"
- emdim = 300
- hidim = 300
  -nlayers = 2
  -use_seed = True
  -use_input = True
  -use_hidden = False
  -use_gated = False
  -use_ch = False
  -use_he = True
  -lr = 0.001
  -decay_factor = 0.1
  -dropout = 0.3
  -dropouth = 0.1
  -dropouti = 0.1
  -dropoute = 0.1
  -wdrop = 0.2
  -wdecay = 1.2e-6
  -alpha = 0
  -beta = 1
  -exp_dir = ".../checkpoints/"
  -w2v_weights = ".../data/processed/embedding.pkl"
  -fix_embeddings = True
  -cuda = False
  
# Generation
  
