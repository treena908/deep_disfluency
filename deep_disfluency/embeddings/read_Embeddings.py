import os
import numpy as np
# import gensim
# from gensim.models import KeyedVectors
# from deep_disfluency.load.load import load_tags
# from deep_disfluency.embeddings.load_embeddings import populate_embeddings
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(THIS_DIR + "/../../")
path=THIS_DIR+'/../experiments/041/epoch_16'
filename='embeddings.npy'
d_matrix = np.load(path + "/" +
                       filename)
print(d_matrix.shape)
print(d_matrix[0])
# word_path = os.path.dirname(os.path.realpath(__file__)) +\
#             "/../data/tag_representations/{}.csv".format('DB_word_rep')
# word_to_index_map = load_tags(word_path)
#
# pretrained = KeyedVectors.load('DB_clean_50')
# emb = populate_embeddings(50,
#                                   len(word_to_index_map.items()),
#                                   word_to_index_map,
#                                   pretrained)
# print({0},{1}.format('vocab len',len(word_to_index_map)))
# print(pretrained)
# print ("emb shape"+ str(pretrained[pretrained.wv.index2word[0]].shape))
# for i in range(0, len(pretrained.wv.index2word)):
#     print(pretrained.wv.index2word[i])
# print(i)
