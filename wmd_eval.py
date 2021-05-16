from nltk.corpus import stopwords
from nltk import download
download('stopwords')  # Download stopwords list.
stop_words = stopwords.words('english')
import gensim.downloader as api
model = api.load('word2vec-google-news-300')
model.init_sims(replace=True)

def preprocess(sentence):
    return [w for w in sentence.lower().split()]

gen_list = []
doc1 = open("/home/cantors2/Documents/xwordPytorch/data/eval/OREO_generated.txt","r")
for i in doc1.readlines():
    gen_list.append(preprocess(i.strip()))
    
data_list = []
doc2 = open("/home/cantors2/Documents/xwordPytorch/data/eval/OREO_database.txt","r")
for i in doc2.readlines():
    data_list.append(preprocess(i.strip()))

dist = {}
for i in gen_list:
    total = []
    for j in data_list:
        if i != j:
            distance = model.wmdistance(i, j)
            total.append(distance)
    dist[str(i)] = sum(total)/len(total)

l = []    
for item in dist.items():
    l.append(item[-1])
#for item in dist.items():
    #print(item)

print(sum(l)/len(l))



