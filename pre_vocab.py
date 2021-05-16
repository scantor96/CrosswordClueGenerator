import time
from utils.datasets import Vocabulary
from utils.util import get_time_dif, read_data, read_hypernyms
import os
from string import punctuation

defs = os.path.join(".../data/defs")
voc = Vocabulary()
char_voc = Vocabulary()
start_time = time.time()
print("Start build the vocabulary at {}".format(time.asctime(time.localtime(start_time))))
for filepath in os.listdir(defs):
    data = read_data(filepath)
    for elem in data:
        elem1 = elem[0].strip(punctuation)
        voc.add_token(elem1)
        char_voc.token_maxlen = max(len(elem[0]), char_voc.token_maxlen)
        for c in elem[0]:
            char_voc.add_token(c)
        definition = elem[1]
        for d in definition:
            voc.add_token(d)

hypm = ".../data/bag_of_hypernyms.txt"
if hypm is not None:
    _, hypm_token = read_hypernyms(hypm)
    for h in hypm_token:
        voc.add_token(h)
voc.save(".../data/processed/vocab.json")
char_voc.save(".../data/processed/char_vocab.json")
time_dif = get_time_dif(start_time)
print("Finished! Build vocabulary time usage:", time_dif)
