"""Methods for:

--creation of vocabulary files from training corpora with index for each word
--creation of files with the disfluency tags and their index.
"""
import argparse
import re
import sys
import os
from csv import writer
from collections import defaultdict
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(THIS_DIR + "/../..")
from deep_disfluency.load.load import load_tags
from feature_utils import load_data_from_corpus_file
from feature_utils import load_data_from_disfluency_corpus_file
from feature_utils import get_tags
from feature_utils import concat_all_data_all_speakers
from feature_utils import sort_into_dialogue_speakers

from deep_disfluency.utils.tools import convert_to_simple_label

def load_DB_words(DB_wordrepfilepath):
    words=[]
    f=open(DB_wordrepfilepath)
    for line in f:
        l = line.strip('\n').split(",")
        # if l[1].startswith('a'):
        #     print('before deleting...')
        #     print(l[1].strip())
        words.append(l[1])

    return words

def create_word_representation_swdb_DB(swdb_wordrepfilepath,DB_wordrepfilepath,swbd_DB_words,

                                      min_occurrences=1,
                                      word_rep_type="one-hot"):
    """Creates a word rep file from a training data file supplied or a
    plain text file. Has to occur at least min_occurrences times
    else set to the unknown token automatically externally.
    """
    # tag_dict = load_tags(swdb_wordrepfilepath)  # tag and the number of times it occur
    word_swdb=load_DB_words(swdb_wordrepfilepath)  # tag and the number of times it occur
    words=load_DB_words(DB_wordrepfilepath)
    remove=[]
    print('db vocab %d'%len(words))
    print("creating word  tag file...")
    i = 0




    for id,tag_sequence in enumerate(words):



        # if tag_sequence in tag_dict.keys() :
        if tag_sequence in word_swdb:

            i += 1


            remove.append(tag_sequence)
    for item in remove:
        words.remove(item)


    print('remove count %d'%i)
    start=9068

    vocab_swdb = [str(i) + "," + str(word_swdb[i])
             for i in range(0, len(word_swdb))]
    # vocab_swbd = [key for key in sorted(tag_dict.keys())]
    vocab_DB = [str( start+i) + "," + str(words[i])
                for i in range(0, len(words))]
    final_vocab = vocab_swdb + vocab_DB
    # final_vocab=[str(i) + "," + str(vocab_swbd[i])
    #          for i in range(0, len(vocab_swbd))]+vocab_DB
    print(len(final_vocab))
    tagstring = "\n".join(final_vocab)
    print("vocab_size", len(final_vocab))
    myfile = open(swbd_DB_words, "w")
    myfile.write(tagstring)
    myfile.close()
    print("word or pos tag file complete.")

def create_word_or_pos_representation(wordrepfilepath,
                                      words_or_pos,
                                      min_occurrences=1,
                                      word_rep_type="one-hot"):
    """Creates a word rep file from a training data file supplied or a
    plain text file. Has to occur at least min_occurrences times
    else set to the unknown token automatically externally.
    """
    tag_dict = defaultdict(int)  # tag and the number of times it occur
    print
    "creating word or pos tag file..."
    i = 0
    for tag_sequence in words_or_pos:
        i += 1
        # print i
        # print len(tag_sequence)
        for a_tag in tag_sequence:
            tag_dict[a_tag] += 1
    vocab_allowed = [key for key in sorted(tag_dict.keys()) if
                     tag_dict[key] >= min_occurrences] + ["<unk>"]
    vocab = [str(i) + "," + str(vocab_allowed[i])
             for i in range(0, len(vocab_allowed))]
    tagstring = "\n".join(vocab)
    print
    "vocab_size", len(vocab)
    myfile = open(wordrepfilepath, "w")
    myfile.write(tagstring)
    myfile.close()
    print
    "word or pos tag file complete."


def create_tag_representations(tag_rep_filepath, tags,ids,words,labels,
                               representation="disf1",tag_corpus_file=None,limit=8):
    """Create the tag files for a given corpus f with a given
    representation type.

    Keyword arguments:
    tag_rep_file --  file to write the tag rep too.
    Note this must be a training file.
    tags -- list of lists of tags (training)
    representation -- string showing the type of tagging system,
    1=standard, 2=rm-N values where N does not count intervening edit terms
    3=same as 2 but with a 'c' tag after edit terms have ended.
    """
    print('ekhane')
    tag_dict = defaultdict(int)  # tag and the number of times it occurs
    print"creating tag file:", representation, "..."
    # print len(tags)
    if tag_corpus_file:
        print"(and creating tag corpus file)"
        tag_corpus_file = open(tag_corpus_file, "w")

    i = 0
    tag_corpus = ""
    for tag_sequence,id,word,label in zip(tags,ids,words,labels):
        i += 1
        # print i
        # print len(tag_sequence)
        for a_tag,w in zip(tag_sequence,word):
            tag = ""
            subtags = get_tags(a_tag)
            if "disf" in representation:
                for t in subtags:
                    if not re.search(r'<[ct]*/>', t) \
                            and not re.search(r'<diact type="[^\s]*"/>', t) \
                            and not re.search(
                        r'<speechLaugh/>|<laughter/>', t) \
                            and not re.search(r'<speaker floor="[^\s]*"/>', t):
                        if "<speaker" in t:
                            print
                            "WARNING speaker getting through"
                        tag += t
            if "disf" in representation and "simple" in representation:
                tag = convert_to_simple_label(tag, "disf1")
            if "dact" in representation:
                m = re.search(r'<diact type="[^\s]*"/>', a_tag)
                if m:
                    tag += m.group(0)
            if "laugh" in representation:
                m = re.search(r'<speechLaugh/>|<laughter/>', a_tag)
                if m:
                    tag += m.group(0)
                else:
                    tag += "<nolaughter/>"
            if "uttseg" in representation:
                m = re.search(r'<[ct]*/>', a_tag)
                if m:
                    tag += m.group(0)
                else:
                    if "<laugh" in a_tag:
                        continue
                    print
                    "No utt seg found", a_tag
                    continue
            if tag == "":
                if "<laugh" not in a_tag:
                    print
                    "warning no tag", a_tag
                continue
            if "interactive" in representation:
                if "speaker" in tag:
                    print
                    "in tag already", a_tag, tag
                m = re.search(r'<speaker floor="[^\s]*"/>', a_tag)
                if m:
                    # if "<speaker" in tag:
                    tag += m.group(0)
            if ("uttseg" not in representation) and "t/>" in a_tag:
                # non segmented mode
                if "<speaker" in a_tag:
                    # only add tag at end as a single tag if interactive
                    tag_corpus += tag + ","
                else:
                    tag_corpus += tag + "\n"
            # do segmentation last as it might not be segmented
            if tag == '<e/><rm-1/><rpEndSub/>' or tag == '<e/><rm-2/><rpEndSub/>' or tag == '<e/><rpEndDel/>' :
                print('print id...')
                print(id)
                print(tag)
                print(w)
                # print(word)
                # print(label)
            tag_dict[tag] += 1
            if ("uttseg" not in representation) and "<speaker" in a_tag:
                continue  # i.e. if interactive treat as a single tag
            if ("uttseg" not in representation):
                m = re.search(r'<[ct]*/>', a_tag)
                if m and "t/>" in m.group(0):
                    tag_corpus = tag_corpus.strip(",") + "\n"
                    continue

            tag_corpus += tag + ","

        tag_corpus = tag_corpus.strip(",") + "\n"
        # new line separated dialogue/speakers
    if tag_corpus_file:
        tag_corpus_file.write(tag_corpus)
        tag_corpus_file.close()
    print (tag_dict)
    tagstring = "\n".join([str(i) + "," + str(sorted(tag_dict.keys())[i])
                           for i in range(0, len(tag_dict.keys()))])
    tag_rep_file = open(tag_rep_filepath, "w")
    tag_rep_file.write(tagstring)
    tag_rep_file.close()
    print
    "tag file complete."
def remove_duplicate(swdb_DB_word_rep_file):
    words=load_DB_words(swdb_DB_word_rep_file)
    converted_set = set(words)

if __name__ == '__main__':
    tag_rep_dir=THIS_DIR + "/../data/tag_representations"
    swbd_word=tag_rep_dir+"/swbd_word_rep_041.csv"
    DB_word = tag_rep_dir+"/DB_word_rep.csv"
    swdb_DB_word_rep_file = tag_rep_dir+ "/swbd_DB_word_rep_update.csv"

    create_word_representation_swdb_DB(swbd_word,DB_word,swdb_DB_word_rep_file,min_occurrences=1)
    # remove_duplicate(swdb_DB_word_rep_file)
    # parser = argparse.ArgumentParser(description='Creates tag sets for \
    #                                  labels.')
    # data_file = '../data/disfluency_detection/switchboard/' + \
    #             'swbd_disf_train_1_partial_data_timings.csv'
    # parser.add_argument('-i', action='store', dest='corpusFile',
    #                     default=data_file,
    #                     help='location of the corpus from which to\
    #                     generate the tag sets')
    # parser.add_argument('-tag', action='store', dest='tagFolder',
    #                     default="../data/tag_representations",
    #                     help='location to store the tag files to\
    #                     tag index mapping')
    #
    # args = parser.parse_args()
    #
    # corpus_file = args.corpusFile
    # print("creating tag files. Using file:")
    # print(corpus_file)
    #
    # word_rep_file = args.tagFolder + "/DB_word_rep.csv"
    # pos_rep_file = args.tagFolder + "/DB_pos_rep.csv"
    # disf_tag_rep_file = args.tagFolder + "/DB_disf1_tags.csv"
    # joint_tag_rep_file = disf_tag_rep_file
    #
    # if "timings" in args.corpusFile:
    #     dialogues = load_data_from_corpus_file(corpus_file,
    #                                            convert_to_dnn_format=False)
    #     _, words, pos, _, labels = concat_all_data_all_speakers(dialogues,
    #                                                             divide_into_utts=
    #                                                             not args.uttSeg,
    #                                                             convert_to_dnn_format=
    #                                                             True)
    #
    # else:
    #     print('creating tagset')
    #     IDs, timings, seq, pos_seq, targets = \
    #         load_data_from_disfluency_corpus_file(
    #             corpus_file,
    #             convert_to_dnn_format=True)
    #     dialogues = sort_into_dialogue_speakers(IDs, timings, seq, pos_seq,
    #                                             targets)
    #     ids=[x[0] for x in dialogues]
    #     words = [x[2] for x in dialogues]
    #     pos = [x[3] for x in dialogues]
    #     labels = [x[4] for x in dialogues]
    #
    # create_tag_representations(disf_tag_rep_file, labels,ids,words,labels,
    #                            representation="disf1",
    #                            tag_corpus_file=disf_tag_rep_file.
    #                            replace("tags", "tag_corpus"))
    #
    #
    # create_word_or_pos_representation(word_rep_file, words,
    #                                   min_occurrences=2)
    # create_word_or_pos_representation(pos_rep_file, pos,
    #                                   min_occurrences=2)
