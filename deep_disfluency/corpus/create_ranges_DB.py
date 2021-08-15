import pandas as pd
import numpy as np
import random
import os
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
def write_dict(path,dictionary):
    print('start writing')
    for ind, key in enumerate(dictionary.keys()):

        with open(path+'/DB_disf_'+key+'_1_ranges.text', 'w') as f:
            print('key: '+key)
            print('len: '+str(len(dictionary[key])))
            for line in dictionary[key]:
                f.write(line)
                f.write('\n')
            f.close()
    print('writing complete')
def create_ranges(path):
    print('start partition')
    range_dir = THIS_DIR + \
                '/../data/disfluency_detection/DB_divisions_disfluency_detection'
    df=pd.read_csv(path)
    dictionary={'train':[],'heldout':[],'test':[]}
    percentage=[.8,.1,.1]
    filenames=df.filename.unique()
    
    for ind,key in enumerate(dictionary.keys()):
        count=0
       
        print('key: '+key)
        print(len(filenames))
        
        
        while count<=percentage[ind]*len(df) and len(filenames)>0: 
          
          


            rand_int=random.randint(0, len(filenames)-1)

            if filenames[rand_int] not in dictionary['train'] and filenames[rand_int] not in dictionary['heldout'] and filenames[rand_int] not in dictionary['test']:
                dictionary[key].append(filenames[rand_int])
                rows=df[df['filename']==filenames[rand_int]]

                count+=len(rows)
                filenames = np.delete(filenames, rand_int)

        print('count: '+str(count))
    print('stop partition')

    write_dict(range_dir,dictionary)
if __name__ == '__main__':


    create_ranges(THIS_DIR+'/../../DisfluencyProject/data_w_pos.csv')

