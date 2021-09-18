import os
import numpy as np
from deep_disfluency.utils.tools import \
    dialogue_data_and_indices_from_matrix
def evaluate_fast_from_matrices(self, validation_matrices, tag_file,
                                idx_to_label_dict):
    output = []
    true_y = []

    for v in validation_matrices:
        if v is None:
            return {}
        words_idx, pos_idx, extra, y, indices = v
        if extra:
            output.extend(self.model.classify_by_index(words_idx, indices,
                                                       pos_idx,
                                                       extra))
        else:
            output.extend(self.model.classify_by_index(words_idx, indices,
                                                       pos_idx))
        true_y.extend(y)
    p_r_f_tags = precision_recall_fscore_support(true_y,
                                                 output,
                                                 average='macro')
    tag_summary = classification_report(
        true_y, output,
        labels=[i for i in xrange(len(idx_to_label_dict.items()))],
        target_names=[
            idx_to_label_dict[i]
            for i in xrange(len(idx_to_label_dict.items()))
        ]
    )
    print(tag_summary)
    results = {"f1_rmtto": p_r_f_tags[2], "f1_rm": p_r_f_tags[2],
               "f1_tto1": p_r_f_tags[2], "f1_tto2": p_r_f_tags[2]}

    results.update({
        'f1_tags': p_r_f_tags[2],
        'tag_summary': tag_summary
    })
    return results
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
validation_dialogues_filepath = THIS_DIR + '/../data/disfluency_detection/feature_matrices/heldout'
validation_matrices = [np.load(
                                    validation_dialogues_filepath + "/" + fp)
                               for fp in os.listdir(
                                validation_dialogues_filepath)]

validation_matrices = [dialogue_data_and_indices_from_matrix(
                          d_matrix,
                          0,
                          pre_seg=True,
                          window_size=3,
                          bs=2,
                          tag_rep=self.args.tags,
                          tag_to_idx_map=self.tag_to_index_map,
                          in_utterances=self.args.utts_presegmented)
                       for d_matrix in validation_matrices
                       ]