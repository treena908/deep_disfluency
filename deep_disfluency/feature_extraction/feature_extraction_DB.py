# Script which calls different scripts for differing feature extraction.
# See the top-level README.md to see where the raw data should be placed.
#
# Required (language model features derivable):
#     Transcript raw data : disfluency detection corpus (already generated by
#     deep_disfluency.corpus.DisfluencyCorpusCreator.py on swda data
# Optional (if using word timing):
#     Transcript raw data: word timings from Mississippi Swbd files (download)
# Optional (if using audio features and/or ASR):
#     Audio raw data : .sph (or .wav) files (purchase from LDC)
#     OpenSmile (software) + .conf file
# Optional (if using ASR results):
#     IBM Watson ASR account
import argparse
import sys
import subprocess
import os


def extract_features_DB(args):
    corpusName = args.divisionFile[args.divisionFile.rfind("/") + 1:].\
        replace("_ranges.text", "")


    corpus_filename = corpusName + "_data"


    if args.newTags:
        # create the tag representations (normally from the training data
        # not allowed to look into unseen tags in the test/dev set
        # assuming the tags we created for DB
        c = [
            sys.executable,
            os.path.dirname(os.path.realpath(__file__)) +
            '/create_tag_files_DB.py',
            '-i', args.corpusLocation + "/" + corpus_filename,
            '-tag', args.tagFolder,
            ]

        subprocess.call(c)
    c = [
        sys.executable,
        os.path.dirname(os.path.realpath(__file__)) +
        '/save_feature_matrices.py',
        '-i', args.corpusLocation + "/" + corpus_filename,
        '-m', args.matrixFolder,
        '-w', args.tagFolder + "/DB_word_rep.csv",
        '-p', args.tagFolder + "/DB_pos_rep.csv",
        '-tag', args.tagFolder + "/DB_disf1_tags.csv"
    ]

    subprocess.call(c)












if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature extraction for\
    disfluency and other tagging tasks from disfluency detection corpora and\
    raw data.')
    parser.add_argument('-i', action='store', dest='corpusLocation',
                        default='../data/disfluency_detection',
                        help='location of the disfluency\
                        detection corpus folder')
    parser.add_argument(
        '-m', action='store',
        dest='matrixFolder',
        default='../data/disfluency_detection/feature_matrices',
        help='location of the disfluency annotation csv files'
                        )
    parser.add_argument('-f', action='store', dest='divisionFile',
                        default='../data/disfluency_detection/\
                        swda_divisions_disfluency_detection/\
                        SWDisfTrain_ranges.text',
                        help='location of the file listing the \
                        files used in the corpus')

    parser.add_argument('-tag', action='store', dest='tagFolder',
                        default=None,
                        help='location of the folder with the tag to\
                        tag index mapping')
    parser.add_argument('-new_tag', action='store_true', dest='newTags',
                        default=False,
                        help='Whether to save a new tag set generated from\
                        the data set to the tag folder.')
    parser.add_argument('-pos', action='store', dest='posTagger',
                        default=None, help='A POSTagger to tag the data.\
                        If None, Gold POS tags assumed.')

    args = parser.parse_args()
    extract_features_DB(args)
