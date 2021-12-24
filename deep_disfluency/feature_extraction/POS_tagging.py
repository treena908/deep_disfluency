from __future__ import division
import os
import sys
from copy import deepcopy
from collections import Counter
from nltk.tag import CRFTagger
from feature_utils import count_tags,get_tags

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(THIS_DIR + "/../..")
from deep_disfluency.feature_extraction.feature_utils import\
    load_data_from_disfluency_corpus_file
from deep_disfluency.feature_extraction.feature_utils import\
     sort_into_dialogue_speakers


# Set the variables according to what you want to do
TRAIN = False
TEST = False
TAG_ASR_RESULTS = False
db_test=False
TAG=True


TAGGER_PATH = "crfpostagger"  # path to the tagger you want to train/apply

# Train and test from disfluency detection format files:
# DISF_DIR = THIS_DIR + "/../data/disfluency_detection/switchboard"
FILE_DIR=THIS_DIR + "/../../DisfluencyProject"
DISF_DIR = THIS_DIR + "/../data/disfluency_detection/DB"
DISFLUENCY_TEST_FILES = [
                    DISF_DIR + "/swbd_disf_train_1_partial_data.csv",
                    DISF_DIR + "/swbd_disf_heldout_1_partial_data.csv",
                    DISF_DIR + "/swbd_disf_train_1_partial_data.csv"
                    ]
DISFLUENCY_TEST_FILES = [
                    DISF_DIR + "/DB_disf_test_1_data.csv",
                    ]
disf_train="DB_disf_test_1_data.csv"

# ASR results from increco outputs
ASR_DIR = THIS_DIR + "/../data/asr_results/"
INC_ASR_FILES = [ASR_DIR + "SWDisfTest_increco.text",
                 ASR_DIR + "SWDisfHeldout_increco.text"]

# The tags for which an entity tag is added to the word
PROPER_NAMES = ["NNP", "NNPS", "CD", "LS", "SYM", "FW"]
informal_contraction=['gonna','gotta','wanna','lemme','dunno','kinda','sorta','gotcha','betcha','mighta','shoulda',
                        'coulda','woulda','musta','couldna','shouldna','wouldna','outta','cuppa','sorta','lotta',
                        'needa','hafta','hasta','oughta','useta','gimme','tellem','cannot','hafta','hasta','supposeta',
                        ]

def disf_tags_from_easy_read(text):
    """List of disfluency tags from the inline easy read marked up utterances
    """
    tags = []
    for w in text.split():
        tags.append(w[:w.rfind(">") + 1])
    return [tag.replace("_", " ") for tag in tags]

def strip_disf_tags_from_easy_read(text):
    """List of strings (words or POS tags) without the disfluency markup
    """
    words = []
    for w in text.split(" "):
        words.append(w[w.rfind(">") + 1:])
    return words

def get_edit_terms_from_easy_read(text, postext):
    """Outputs tuples of each string of consecutive edit terms and their POS"""
    words = strip_disf_tags_from_easy_read(text)
    pos = strip_disf_tags_from_easy_read(postext)
    tags = disf_tags_from_easy_read(text)
    current_edit_term = ""
    current_pos_edit_term = ""
    # a list of tuples of (edit term strings, POS tags of that string)
    edit_terms = []
    for t in range(0, len(tags)):
        tag = tags[t]
        if "<e" in tag or "<i" in tag:
            current_edit_term += words[t] + " "
            current_pos_edit_term += pos[t] + " "
        elif not current_edit_term == "":  # we've built up a string, save it
            edit_terms.append(
                (current_edit_term.strip(), current_pos_edit_term.strip()))
            current_edit_term = ""
            current_pos_edit_term = ""
    if not current_edit_term == "":  # flush
        edit_terms.append(
            (current_edit_term.strip(), current_pos_edit_term.strip()))
    return edit_terms
def write_edit_term_corpus(testcorpus, targetfilename, debug=False):
    """Write a file cleaned of reparanda and edit terms from a test corpus.

    Keyword Arguments:
    testcorpus -- a string separated by newline markers which
    has all the disfluency markup.
    targetfilename -- a string giving the location of
    the edit term filled corpus.
    """
    print("Writing edit term corpus...")
    edit_term_corpus = open(targetfilename , "w")
    for line in testcorpus.split("\n"):
        if line == "":
            continue
        split = line.split(",")
        # no need to write the indices to source data
        if split[0] == "REF":
            continue
        elif split[0] == "POS":
            pos = split[1]
        else:
            uttref = split[0]
            text = split[1]
            continue
        editterm_examples = get_edit_terms_from_easy_read(text, pos)
        if editterm_examples != []:
            for my_editterm, my_poseditterm in editterm_examples:
                edit_term_corpus.write(
                    uttref + "," + my_editterm + "\nPOS," +
                    my_poseditterm + "\n")
    edit_term_corpus.close()
    print("done")
    return
def write_clean_corpus(testcorpus, targetfilename):
    """Write a file cleaned of reparanda and edit terms from a test corpus.

    Keyword Arguments:
    testcorpus -- a string separated by newline markers which
    has all the disfluency markup.
    targetfilename -- a string giving the location of the cleaned corpus.
    """
    print("Writing clean corpus...")
    clean_corpus = open(targetfilename , "w")
    for line in testcorpus.split("\n"):
        # print line
        if line == "":
            continue
        split = line.split(",")
        # no need to write the indices to source data
        if split[0] == "REF":
            continue
        elif split[0] == "POS":
            pos = split[1]
        else:
            uttref = split[0]
            text = split[1]
            continue

        words = strip_disf_tags_from_easy_read(text)
        pos_tags = strip_disf_tags_from_easy_read(pos)
        disfluencies = disf_tags_from_easy_read(text)
        clean_word_string = ""
        clean_pos_string = ""
        for i in range(0, len(disfluencies)):
            if "<e" in disfluencies[i] or "<rm" in disfluencies[i] \
                    or "<i" in disfluencies[i]:
                continue
            clean_word_string += words[i] + " "
            clean_pos_string += pos_tags[i] + " "
        clean_word_string = clean_word_string.strip()
        clean_pos_string = clean_pos_string.strip()
        if clean_word_string == "":
            continue
        clean_corpus.write(
            uttref + "," + clean_word_string + "\nPOS," +
            clean_pos_string + "\n")
    clean_corpus.close()
    print ("done")
    return
def easy_read_disf_format(words, tags):
    """Easy read style inline disfluency tagged string."""
    final_tags = []
    for i in range(0, len(words)):
        word=words[i]
        if '&-' in word:
            word=word.replace('&-','')
        elif '&' in word:
            word = word.replace('&', '')
        elif '\'' in word:

            word = word.replace('\'', '')


        final_tags.append("".join([tags[i].replace(" ", "_"), word.lower()]))
    return " ".join(final_tags)
def easy_read_disf_format_pos(words, tags):
    """Easy read style inline disfluency tagged string."""
    final_tags = []
    for i in range(0, len(words)):
        final_tags.append("".join([tags[i].replace(" ", "_"), words[i]]))
    return " ".join(final_tags)
def detection_corpus_format(uttRef, words, pos, tags, indices):
    """Replace blanks with fluent <f/> tags and outputs tag separated."""
    for i in range(0, len(tags)):
        if tags[i] == "":
            tags[i] = "<f/>"

    final_string = "\t".join(
        [uttRef, indices.pop(0), words.pop(0), pos.pop(0), tags.pop(0)]) + "\n"


    for i in range(0, len(tags)):
        final_string += "\t".join(["", indices[i],
                                   words[i], pos[i], tags[i]]) + "\n"
    return final_string.rstrip("\n")


if TRAIN:
    ct = CRFTagger()  # initialize tagger
    dialogue_speakers = []
    for disf_file in DISFLUENCY_TRAIN_FILES:
        IDs, mappings, utts, pos_tags, labels = \
            load_data_from_disfluency_corpus_file(disf_file)
        dialogue_speakers.extend(sort_into_dialogue_speakers(IDs,
                                                             mappings,
                                                             utts,
                                                             pos_tags,
                                                             labels))
    word_pos_data = {}  # map from the file name to the data
    for data in dialogue_speakers:
        dialogue, a, b, c, d = data
        word_pos_data[dialogue] = (a, b, c, d)

    training_data = []
    for speaker in word_pos_data.keys():
        sp_data = []
        prefix = []
        for word, pos in zip(word_pos_data[speaker][1],
                             word_pos_data[speaker][2]):
            prefix.append(word.replace("$unc$", ""))
            sp_data.append((unicode(word.replace("$unc$", "")
                                    .encode("utf8")),
                            unicode(pos.encode("utf8"))))
        training_data.append(deepcopy(sp_data))
    print "training tagger..."
    ct.train(training_data, TAGGER_PATH)

if db_test:
    print
    "testing tagger..."
    ct = CRFTagger()  # initialize tagger
    ct.set_model_file(TAGGER_PATH)
    print(ct.tag([unicode(w) for w in "uh and then there's a tree that doesn't have a trunk".split()]))
    print(ct.tag([unicode(w) for w in "uh-huh yeah yeah".split()]))
def make_utterance(u):
    utterance=""
    for i, utt in enumerate(u):
        if '&-' in utt:
            utt = utt.replace('&-', '')


        elif '&' in utt:

            utt=utt.replace('&','')
        elif '\'' in utt:

            utt = utt.replace('\'', '')



        if i<len(u)-1:
            utterance+=utt.lower()+" "
        else:
            utterance += utt.lower()
    return utterance
unk=False
def update_label(label):
    for t in get_tags(label):
        if '<e/>' not in t or '<i' not in t:
            label=label.replace(t,"")



def make_pos_list(pos):
    final_pos=[]
    final_word=[]
    for elem in pos:
        if elem[1] in PROPER_NAMES and elem[0].lower() not in informal_contraction:
            # print('proper')
            w='$unc$'+elem[0].lower()
            unk=True
            # print(w)
        else:
            w = elem[0].lower()
        final_pos.append(elem[1])
        final_word.append(w)
    return final_word,final_pos

if TAG:

    print"testing disf_tagger..."
    ct = CRFTagger()  # initialize tagger
    ct.set_model_file(TAGGER_PATH)
    dialogue_speakers = []
    writeFile = True
    writecleanFile = True
    writeeditFile = True

    corpus = ""
    easyread_corpus = ""
    for disf_file in DISFLUENCY_TEST_FILES:
        IDs, mappings, utts, pos_tags, labels = \
            load_data_from_disfluency_corpus_file(disf_file)
        for i, m, u, p, label in zip(IDs, mappings, utts, pos_tags, labels):
            new_label=[]
            for l in label:
                count=count_tags(l)
                if '<e/>' in l and '<i' in l and count>2:
                    new_l=update_label(l)
                    new_label.append(new_l)
                else if '<e/>' in l and '<i' not in l and '<rpndel' in l:
                    new_l = update_label(l)
                    new_label.append(new_l)
                else:
                    new_label.append(new_l)

            break
        break

        if len(new_label) != len(u):
            print('len mismatch')
            pass
        wordstring = easy_read_disf_format(
            u, new_l)
        posstring = easy_read_disf_format_pos(
            p, new_l)
        indexstring = ""

        indexstring = " ".join(m)
        easyread_format = i + "," + wordstring + \
                          "\nPOS," + posstring + "\n" + "REF," + indexstring

        corpus_format = detection_corpus_format(
            i, deepcopy(new_word),
            deepcopy(new_pos),
            deepcopy(new_label), deepcopy(list(m)))
        corpus += corpus_format + "\n"

        easyread_corpus += easyread_format + "\n"

        if writeFile:
            disffile = open(disf_train, "w")
            disffile.write(corpus)
            disffile.close()
        if writecleanFile:
            write_clean_corpus(easyread_corpus, disf_train.replace("_data.csv", "_clean.text"))

        if writeeditFile:
            write_edit_term_corpus(easyread_corpus, disf_train.replace("_data.csv", "_edit.text"))
        dialogue_speakers.extend(sort_into_dialogue_speakers(IDs,
                                                             mappings,
                                                             utts,
                                                             new_pos,
                                                             labels))

if TEST:
    print "testing tagger..."
    ct = CRFTagger()  # initialize tagger
    ct.set_model_file(TAGGER_PATH)
    dialogue_speakers = []
    writeFile = True
    writecleanFile=True
    writeeditFile=True

    corpus=""
    easyread_corpus=""
    for disf_file in DISFLUENCY_TEST_FILES:
        IDs, mappings, utts, pos_tags, labels = \
            load_data_from_disfluency_corpus_file(disf_file)
        for i,m,u,p,l in zip(IDs,mappings,utts,pos_tags,labels):


            utterance=make_utterance(u)
            new_word,new_pos=make_pos_list((ct.tag([unicode(w) for w in utterance.split()])))


            if len(new_pos)!=len(u):
                print('len mismatch')
                pass
            wordstring = easy_read_disf_format(
                u, l)
            posstring = easy_read_disf_format_pos(
                p, l)
            indexstring = ""

            indexstring = " ".join(m)
            easyread_format = i + "," + wordstring + \
                              "\nPOS," + posstring + "\n" + "REF," + indexstring
            # print('transname: '+ uttref)
            # print(list(new_pos))
            # corpus_format = detection_corpus_format(
            #     i, deepcopy(list(u)),
            #     deepcopy(new_pos),
            #     deepcopy(list(l)), deepcopy(list(m)))
            corpus_format = detection_corpus_format(
                i, deepcopy(new_word),
                deepcopy(new_pos),
                deepcopy(list(l)), deepcopy(list(m)))
            corpus += corpus_format + "\n"


            easyread_corpus += easyread_format + "\n"

        if writeFile :
            disffile = open(disf_train, "w")
            disffile.write(corpus)
            disffile.close()
        if writecleanFile:
            write_clean_corpus(easyread_corpus, disf_train.replace("_data.csv","_clean.text") )

        if writeeditFile:
            write_edit_term_corpus(easyread_corpus, disf_train.replace("_data.csv","_edit.text"))
        dialogue_speakers.extend(sort_into_dialogue_speakers(IDs,
                                                             mappings,
                                                             utts,
                                                             new_pos,
                                                             labels))

    word_pos_data = {}  # map from the file name to the data

    for data in dialogue_speakers:
        dialogue, a, b, c, d = data
        word_pos_data[dialogue] = (a, b, c, d)


    ct.tag([unicode(w) for w in "uh and then there's a tree that doesn't have a trunk".split()])
    # either gather training data or test data
    training_data = []
    for speaker in word_pos_data.keys():
        # print speaker
        sp_data = []
        prefix = []
        predictions = []
        for word, pos in zip(word_pos_data[speaker][1],
                             word_pos_data[speaker][2]):
            prefix.append(unicode(word.replace("$unc$", "")
                                  .encode("utf8")))
            prediction = ct.tag(prefix[-5:])[-1][1]
            sp_data.append((unicode(word.replace("$unc$", "")
                                    .encode("utf8")),
                            unicode(pos.encode("utf8"))))
            predictions.append(prediction)
        training_data.append(deepcopy([(r, h)
                                       for r, h in zip(predictions, sp_data)]))
    # testing
    tp = 0
    fn = 0
    fp = 0
    overall_tp = 0
    overall_count = 0
    c = Counter()
    for t in training_data:
        for h, r in t:
            # print h,r
            overall_count += 1
            hyp = h
            if hyp == "UH":
                if not r[1] == "UH":
                    fp += 1
                else:
                    # print h,r
                    tp += 1
            elif r[1] == "UH":
                fn += 1
            if hyp == r[1]:
                overall_tp += 1
            else:
                c[hyp + "-" + r[1]] += 1
    # print tp, fn, tp
    # p = (tp/(tp+fp))
    # r = (tp/(tp+fn))
    # print "UH p, r, f=", p, r, (2 * p * r)/(p+r)
    # print "overall accuracy", overall_tp/overall_count
    # print "most common errors hyp-ref", c.most_common()[:20]

if TAG_ASR_RESULTS:
    def get_diff_and_new_prefix(current, newprefix, verbose=False):
        """Only get the different right frontier according to the timings
        and change the current hypotheses"""
        if verbose:
            print "current", current
            print "newprefix", newprefix
        rollback = 0
        original_length = len(current)
        for i in range(len(current)-1, -2, -1):
            if verbose:
                print "oooo", current[i], newprefix[0]
            if i == -1 or (float(newprefix[0][1]) >= float(current[i][2])):
                if i == len(current)-1:
                    current = current + newprefix
                    break
                k = 0
                marker = i+1
                for j in range(i+1, len(current)):
                    if k == len(newprefix):
                        break
                    if verbose:
                        print "...", j, k, current[j], newprefix[k]
                        print len(newprefix)
                    if not current[j] == newprefix[k]:
                        break
                    else:
                        if verbose:
                            print "repeat"
                        k += 1
                        marker = j+1
                rollback = original_length - marker
                current = current[:marker] + newprefix[k:]
                newprefix = newprefix[k:]
                break
        return (current, newprefix, rollback)

    print "tagging ASR results..."
    ct = CRFTagger()  # initialize tagger
    ct.set_model_file(TAGGER_PATH)
    # now tag the incremental ASR result files of interest
    for filename in INC_ASR_FILES:
        # always tag the right frontier
        current = []
        right_frontier = 0
        rollback = 0
        newfile = open(filename.replace("increco.",
                                        "pos_increco."),
                       "w")
        a_file = open(filename)
        dialogue = 0
        for line in a_file:
            if "File:" in line:
                dialogue = line
                newfile.write(line)
                current = []
                right_frontier = 0
                rollback = 0
                continue
            if "Time:" in line:
                increment = []
                newfile.write(line)
                continue
            if line.strip("\n") == "":
                if current == []:
                    current = deepcopy(increment)
                else:
                    verb = False
                    current, _, rollback = get_diff_and_new_prefix(
                                                deepcopy(current),
                                                deepcopy(increment),
                                                verb)
                for i in range(right_frontier - rollback, len(current)):
                    test = [unicode(x[0].lower().replace("'", ""))
                            for x in current[max([i-4, 0]):i+1]]
                    # if "4074A" in dialogue:
                    #    print "test", test
                    prediction = ct.tag(test)[-1][1]
                    word = current[i][0].lower().replace("'", "")
                    if prediction in PROPER_NAMES:
                        word = "$unc$" + word
                    start = current[i][1]
                    end = current[i][2]
                    newfile.write("\t".join([str(start),
                                            str(end), word] +
                                            [prediction]) + "\n"
                                  )
                right_frontier = len(current)
                newfile.write(line)
            else:
                spl = line.strip("\n").split("\t")
                increment.append((spl[0], float(spl[1]), float(spl[2])))
        file.close()
        newfile.close()
