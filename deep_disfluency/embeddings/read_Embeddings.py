import gensim
from gensim.models import KeyedVectors

pretrained = KeyedVectors.load('db_clean_50')
print(pretrained)
print ("emb shape"+ str(pretrained[pretrained.wv.index2word[0]].shape))
