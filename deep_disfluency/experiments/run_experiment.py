
# Script to run the experiments described in:

# Julian Hough and David Schlangen.
# Joint, Incremental Disfluency Detection and
# Utterance Segmentation from Speech.
# EACL 2017.

import sys
import subprocess
import os
import urllib
import zipfile
import tarfile
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(THIS_DIR + "/../../")

from deep_disfluency.tagger.deep_tagger import DeepDisfluencyTagger

print('this_dir'+ THIS_DIR)
# The data must been downloaded
# and put in place according to the top-level README
# each of the parts of the below can be turned off
# though they must be run in order so the latter stages work
download_raw_data = False
create_disf_corpus = False
extract_features =False
train_models = True
test_models = False
debug=False

asr = False  # extract and test on ASR results too
partial = False  # whether to include partial words or not

range_dir = THIS_DIR + \
    '/../data/disfluency_detection/DB_divisions_disfluency_detection'
file_divisions_transcripts = [
    ('train', range_dir + '/DB_disf_train_1_ranges.text'),
    # range_dir + '/swbd_disf_train_audio_ranges.text',
    ('heldout', range_dir + '/DB_disf_heldout_1_ranges.text'),
   ('test', range_dir + '/DB_disf_test_1_ranges.text')
]


# the experiments in the EACL paper
# 33 RNN simple tags, disf + utt joint
# 34 RNN complex tags, disf + utt joint
# 35 LSTM simple tags, disf + utt joint
# 36 LSTM complex tags, disf + utt joint
# 37 LSTM simple tags, disf only
# 38 LSTM simple tags, utt only
# 39 LSTM complex tags, disf only
# experiments = [33, 34, 35, 36, 37, 38]
experiments = [44]

# experiments = [35]  # short version for testing
# 1. Download the SWDA and word timings
# if download_raw_data:
#     name = THIS_DIR + '/../data/raw_data/swda.zip'
#     if not os.path.isfile(name):
#         print 'downloading', name
#         urllib.urlretrieve(SWDA_CORPUS_URL, name)
#         zipf = zipfile.ZipFile(name)
#         zipf.extractall(path=SWDA_CORPUS_DIR)
#         zipf.close()
#         print 'extracted at', SWDA_CORPUS_DIR
#
#     name = THIS_DIR + '/../data/raw_data/' + SWBD_TIMINGS_URL.split('/')[-1]
#     if not os.path.isfile(name):
#         print 'downloading', name
#         urllib.urlretrieve(SWBD_TIMINGS_URL, name)
#         tar = tarfile.open(name)
#         tar.extractall(path=SWBD_TIMINGS_DIR)
#         tar.close()
#         print 'extracted at', SWBD_TIMINGS_DIR

# 2. Create the base disfluency tagged corpora in a standard format
"""
for all divisions call the corpus creator
parse c line parameters
Optional arguments:
-i string, path of source data (in swda style)
-t string, target path of folder for the preprocessed data
-f string, path of file with the division of files to be turned into
a corpus
-a string, path to disfluency annotations
-lm string, Location of where to write a clean language\
model files out of this corpus
-pos boolean, Whether to write a word2pos mapping folder
in the sister directory to the corpusLocation, else assume it is there
-p boolean, whether to include partial words or not
-d boolean, include dialogue act tags in the info
"""
if create_disf_corpus:
    print('Creating corpus...')
    write_pos_map = True
    for div, divfile in file_divisions_transcripts:
        c = [sys.executable, THIS_DIR +
             '/../corpus/deep_disfluency_db_corpus_creator.py',
             '-i', THIS_DIR+'/../../DisfluencyProject/',
             '-t', THIS_DIR + '/../data/disfluency_detection/DB/',
             '-f', divfile,

             # '-lm', 'data/lm_corpora',

             ]

        # if write_pos_map:
        #     c.append('-pos')
        #     write_pos_map = False  # just call it once
        subprocess.call(c)
    print('Finished creating corpus.')

# 3. Run the preprocessing and extraction of features for all files
"""
note to get the audio feature extraction to work you need to have
optional arguments are:
i string, path of source disfluency corpus
t string, target path of folder feature matrices in this folder
 (rather than use text files)
f string, path of file with the division of files to be turned into
a corpus of vectors
p boolean, whether to include partial words or not
a string, path to word alignment folder
tag string, path of folder with tag representations
new_tag bool, whether to write new tag representations or use old ones
pos path, path to POS tagger if using one, if None use gold
train_pos bool, whether to train pos tagger or not and put it in pos
u bool, include utterance segmentation tags, derivable from utts
d bool, include dialogue act tags
l bool, include laughter tags on words- either speech laugh on word or
bout
joint bool, include big joint tag set as well as the individual ones
lm string, Location of where to write a clean language\
model files out of this corpus
xlm boolean, Whether to use a cross language model\
training to be used for getting lm features on the same data.
asr boolean, whether to produce ASR results for creation of the
data or not
credentials string, username:password for IBM ASR
audio string, path to open smile for audio features, if None
no audio extraction.
"""
if extract_features:
    print('Extracting features...')
    tags_created = True
    tagger_trained = False
    for div, divfile in file_divisions_transcripts:
        print("divfile :"+divfile)
        c = [sys.executable, THIS_DIR +'/../feature_extraction/feature_extraction_DB.py',
             '-i', THIS_DIR +'/../data/disfluency_detection/DB',
             '-m', THIS_DIR +'/../data/disfluency_detection/feature_matrices/'+div,
             '-f', divfile,

             '-tag', THIS_DIR +'/../data/tag_representations',

             # '-lm', "data/lm_corpora"
             ]

        if not tags_created:
            c.append('-new_tag')
            tags_created = True

        subprocess.call(c)
    print('Finished extracting features.')
# 4. Train the model on the transcripts (and audio data if available)
# NB each of these experiments can take up to 24 hours
systems_best_epoch = {}
if train_models:
    feature_matrices_filepath = THIS_DIR + '/../data/disfluency_detection/feature_matrices/train'

    validation_filepath = THIS_DIR + '/../data/disfluency_detection/feature_matrices/heldout'

    # train until convergence
    # on the settings according to the numbered experiments in
    # experiments/config.csv file
    trained_model = '%03d' % 41
    for exp in experiments:
        disf = DeepDisfluencyTagger(
            config_file=THIS_DIR + "/experiment_configs.csv",
            config_number=exp,
            saved_model_dir=THIS_DIR +
                            '/{0}/epoch_{1}'.format(trained_model, str(16))
        )
        # disf = DeepDisfluencyTagger(
        #     config_file=THIS_DIR + "/experiment_configs.csv",
        #     config_number=exp
        #     )
        exp_str = '%03d' % exp
        e = disf.train_net(
                    train_dialogues_filepath=feature_matrices_filepath,
                    validation_dialogues_filepath=validation_filepath,
                    model_dir=THIS_DIR + '/' + 'DB'+exp_str,
                    tag_accuracy_file_path=THIS_DIR +
                    '/results_DB/tag_accuracies/{}.text'.format(exp_str+"retrain"))
        systems_best_epoch[exp] = e
else:
    # 33 RNN simple tags, disf + utt joint
    # 34 RNN complex tags, disf + utt joint
    # 35 LSTM simple tags, disf + utt joint
    # 36 LSTM complex tags, disf + utt joint
    # 37 LSTM simple tags, disf only
    # 38 LSTM simple tags, utt only
    # 39 LSTM complex tags, disf only
    # Take our word for it that the saved models are the best ones:
    systems_best_epoch[43] = 12  # lstm
    # systems_best_epoch[34] = 37  # RNN (complex tags)
    # systems_best_epoch[35] = 6   # LSTM
    # systems_best_epoch[36] = 15  # LSTM (complex tags)
    # systems_best_epoch[37] = 6   # LSTM (disf only)
    # systems_best_epoch[38] = 8   # LSTM (utt only)

# 5. Test the models on the test transcripts according to the best epochs
# from training.
# The output from the models is made in the folders
# For now all use timing data
# if test_models:
#     print ("testing models...")
#     for exp, best_epoch in sorted(systems_best_epoch.items(),
#                                   key=lambda x: x[0]):
#         for timing_bool in [False]:  # test with and without timing info
#
#             exp_str = '%03d' % exp
#             # load the model
#             disf = DeepDisfluencyTagger(
#                             config_file=THIS_DIR + '/experiment_configs.csv',
#                             config_number=exp,
#                             saved_model_dir=THIS_DIR +
#                             '/{0}/epoch_{1}'.format('DB'+exp_str, best_epoch),
#                             use_timing_data=timing_bool
#                                         )
#             # simulating (or using real) ASR results
#             # for now just saving these in the same folder as the best epoch
#             # also outputs the speed
#             timing_string = '_timings' if timing_bool else ''
#             partial_string = '_partial' if partial else ''
#             for div in ['heldout','test']:
#                 disf.incremental_output_from_file(
#                         THIS_DIR +
#                         '/../data/disfluency_detection/DB/' +
#                         'DB_disf_{0}{1}_1_data.csv'.format(
#                             div, partial_string),
#                         target_file_path=THIS_DIR + '/{0}/epoch_{1}/'.format(
#                             'DB'+exp_str, best_epoch) +
#                         'DB_disf_{0}{1}{2}_data_output_increco.text'
#                         .format(div, partial_string, timing_string)
#                         )
#
if debug:
    feature_matrices_filepath = THIS_DIR + '/../data/disfluency_detection/feature_matrices/train'

    validation_filepath = THIS_DIR + '/../data/disfluency_detection/feature_matrices/heldout'

    # train until convergence
    # on the settings according to the numbered experiments in
    # experiments/config.csv file
    for exp in experiments:
        exp_str = '%03d' % exp
        disf = DeepDisfluencyTagger(
            config_file=THIS_DIR + "/experiment_configs.csv",
            config_number=exp,
            saved_model_dir=THIS_DIR +
                            '/{0}/epoch_{1}'.format('DB' + exp_str, str(12))
            )

        disf.evaluate_result_from_trained_model(

                    validation_dialogues_filepath=validation_filepath,

                    tag_accuracy_file_path=THIS_DIR +
                    '/results_DB/tag_accuracies/{}.text'.format(exp_str+'tesing_micro'))

# 6. To get the numbers run the notebook:
# experiments/analysis/EACL_2017/EACL_2017.ipynb
# The results should be consistent with that in the EACL 2017 paper.