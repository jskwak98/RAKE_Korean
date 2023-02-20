from kiwipiepy import Kiwi
from tqdm import tqdm
from collections import defaultdict
from typing import DefaultDict, Dict
import pickle

from k_rake import K_RAKE
from rake_tfidf import TF_IDF_UPON_RAKE


Word = str
Count = int
Document_ID = int


# register userwords to kiwi tokenizer
kiwi = Kiwi(model_type='sbg')

# TODO : Clarify userword format, also, need to implement os.path exist userword pickle kind of codes
with open("userword", 'rb') as f:
    userword = pickle.load(f)
    for word in userword:
        kiwi.add_user_word(word, tag='NNP', score=30)

# stopwords
with open("rake_stopwords", 'rb') as f:
    stopwords = pickle.load(f)

# whole data, need to specify formats
with open('corpus', 'rb') as f:
    data = pickle.load(f)

# Compute RAKE score while preparing ingredients for TF-IDF
idx_kwd : Dict[Document_ID, Dict] = dict()
idf : DefaultDict[Word, Count] = defaultdict(int)

  
# TODO : Design more efficient way, make reset for K_RAKE class, and reset it after computation
# rather than just creating instance every for loop.
print("Start computing RAKE for documents")
for idx in tqdm(data):
    krake = K_RAKE(stopwords=stopwords, tokenizer=kiwi)
    document = data[idx]['stc']
    kwds = krake.extract_keywords(document)
    idx_kwd[idx] = dict()
    idx_kwd[idx]['kwds'] = kwds
    idx_kwd[idx]['tf'] = krake.phrase_frequency
    for kwd in kwds:
        idf[kwd[1]] += 1


with open("idx_kwd_RAKE", "wb") as f:
    pickle.dump(idx_kwd, f)

with open("idf_RAKE", "wb") as f:
    pickle.dump(idf, f)

print("Saved Computation Result")

# Apply TF-IDF on the key phrases extracted
tfidfrake = TF_IDF_UPON_RAKE(idx_kwd, idf)
print("Computing TF-IDF score for all Key Phrases, It might take a while.")
tfidfrake.compute_tfidf()
rank = tfidfrake.compute_metric()


with open("keyword_ranks", "wb") as f:
    pickle.dump(rank, f)
print("Finished Computation, all results are saved.")
