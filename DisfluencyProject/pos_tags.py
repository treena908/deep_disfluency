import pickle
import pandas as pd
import stanfordnlp
stanfordnlp.download('en')
nlp = stanfordnlp.Pipeline(lang="en", treebank="fr_gsd")
def incompatible(text):
    if not text:
        return True
    return False
def get_pos_tags(text):
    doc = nlp(text)
    tags = doc.sentences[0]
    return tags
def generate_pos_tags():
    df=pd.read_csv('data.csv')
    pos=[]
    for index, row in df.iterrows():
        print(row['cleaned_text_wo_disfluency_markers'])
        print(type(row['cleaned_text_wo_disfluency_markers']))
        parts=row['cleaned_text_wo_disfluency_markers'].split(' ')
        print(parts)
        if incompatible(row['cleaned_text_wo_disfluency_markers']):
            continue
        pos_tags = get_pos_tags(row['cleaned_text_wo_disfluency_markers'])
        pos.append(pos_tags)
        print(pos)
        break
    if len(df)==len(pos):
        df['pos_tag']=pos
    else:
        print('length mismatch')
    df.to_csv("data_w_pos.csv")


generate_pos_tags()