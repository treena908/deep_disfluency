import pandas as pd
import random
import os
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
def write_dict(path,dict):
    for ind, key in enumerate(dict.keys()):

        with open(path+'/DB_disf_'+key+'_1_ranges.text', 'w') as f:
            for line in dict[key]:
                f.write(line)
                f.write('\n')
def create_ranges(path):
    range_dir = THIS_DIR + \
                '/../data/disfluency_detection/DB_divisions_disfluency_detection'
    df=pd.read_csv(path)
    dict={'train':[],'heldout':[],'test':[]}
    percentage=[.8,.1,.1]
    filenames=df.filename.unique()
    for ind,key in enumerate(dict.keys()):
        count=0
        while count<=percentage[ind]*len(df):


            rand_int=random.randint(0, len(filenames))
            if filenames[rand_int] not in dict['train'] and filenames[rand_int] not in dict['heldout'] and filenames[rand_int] not in dict['test']:
                dict[key].append(filenames[rand_int])
    write_dict(range_dir,dict)
if __name__ == '__main__':


    create_ranges(THIS_DIR+'/../../DisfluencyProject/data_w_pos.csv')

