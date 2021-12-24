import numpy
from copy import deepcopy
import os
import sys
import gensim
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(THIS_DIR + "/../../")

def populate_both_embeddings(emb_dim, vocsize, words2index, pretrained_swbd,pretrained_DB):
    """From pre-trained embeddings and matrix dimension generates a random
    matrix first,
    then assigns the appropriate slices from the pre-learned embeddings.
    The words without a corresponding embedding inherit random init.
    """
    # assert pretrained_swbd.layer1_size == emb_dim, str(pretrained_swbd.layer1_size) + \
    #     " " + str(emb_dim)
    assert pretrained_DB.layer1_size == emb_dim, str(pretrained_DB.layer1_size) + \
                                                   " " + str(emb_dim)
    emb = 0.2 * numpy.random.uniform(-1.0, 1.0,
                                     (vocsize+1, emb_dim)).astype('Float32')
    vocab = deepcopy(words2index.keys())

    # print('words')
    # print(words2index.keys())
    print('vocab len')
    print(len(words2index.keys()))
    # db embedding
    for i in range(0, len(pretrained_DB.wv.index2word)):
        word = pretrained_DB.wv.index2word[i]
        if word not in vocab:
            continue
        index = words2index.get(word + '\r')
        if index is None:
            # i.e. no index for this word
            print
            "no such word in vocab for embedding for word:", word
            continue
        # print i, word # returns the word and its index in pretrained
        emb[index] = pretrained_swbd[word]  # assign the correct index
        # vocab.remove(word + '\r')
    #swbd embedding
    for index in range(0, len(pretrained_swbd)):
        emb[index] = pretrained_swbd[index]  # assign the correct index
        # vocab.remove(word+'\r')

    # print len(vocab), "words with no pretrained embedding."
    # for v in vocab:
    #     print v
    return emb
