import gensim
from gensim.models import KeyedVectors
pretrained = KeyedVectors.load('db_clean_50')
print(pretrained)