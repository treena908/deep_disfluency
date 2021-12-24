
# coding: utf-8

# In[24]:

from __future__ import print_function


# In[25]:

from nltk.tag import CRFTagger
# see https://www.nltk.org/_modules/nltk/tag/crf.html for details on training/evaluation


# In[26]:

TAGGER_PATH = "crfpostagger"   # pre-trained POS-tagger


# In[27]:

tagger = CRFTagger()  # initialize tagger
tagger.set_model_file(TAGGER_PATH)


# In[30]:

# try some sentences out- must all be unicode strings- trained on lower case
print(tagger.tag([u"i",u"am", u"gonna", u"talk",u"to",u"you",u"for",u"a",u"little",u"bit"]))
print(tagger.tag([u"it's", u"gonna", u"rain", u"today"]))


# In[31]:

# scaling up as you might get them in text- make sure unicode and lower case
sentences = ["I like revision",
            "I like Natural Language Processing",
             "He has Alzheimer's"]
print(tagger.tag_sents([unicode(word.lower()) for word in s.split()] for s in sentences))


# In[ ]:



