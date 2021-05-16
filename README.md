# CrosswordClueGenerator

# Dependencies
- puzpy 0.2.5 https://github.com/alexdej/puzpy
- PyTorch 1.8.0 https://pytorch.org/
- tqdm 4.50.2 https://pypi.org/project/tqdm/
- NLTK 3.5 https://www.nltk.org/
- gensim 3.8.3 https://pypi.org/project/gensim/
- NumPy 1.19.2 https://numpy.org/install/

# Preprocessing
1. Download Google News Vectors https://code.google.com/archive/p/word2vec/ and save to .../data/word2vec/
For next steps, input appropriate absolute paths wherever a ".../xxxx/..." spot is.
3. Run pre_vocab.py
4. Run pre_wordemb.py
5. Run pre_hypm.py, changing to train/valid/text files each for the defs, save, save_hypm, save_weights variables 
6. Run pre_input_vectors.py

# Training
Run train.py, until it stops training, with the following parameters:
- rnn_type = "LSTM"
- emdim = 300
- hidim = 300
- nlayers = 2
- use_seed = True
- use_input = True
- use_hidden = False
- use_gated = False
- use_ch = False
- use_he = True
- lr = 0.001
- decay_factor = 0.1
- dropout = 0.3
- dropouth = 0.1
- dropouti = 0.1
- dropoute = 0.1
- wdrop = 0.2
- wdecay = 1.2e-6
- alpha = 0
- beta = 1
- exp_dir = ".../checkpoints/"
- w2v_weights = ".../data/processed/embedding.pkl" (sent by email)
- fix_embeddings = True
- cuda = False
  
# Generation
Run generate.py with the most recent model in ".../checkpoints/". If you haven't trained on your own, run with the model provided (by email).
Upon running, the program will request an input puzzle. Copy the absolute path from ".../data/test.puz" and paste into input.

# Evaluation
Run wmd_eval.py for each set of words in ".../data/eval/"
In line 23, a for-loop begins. For each test word, run it once with the first line being "for i in data_list" (this provides you with the WMD for the database clues) and once with the first line being "for i in gen_list" (this provides you with the WMD for the generated clues).


