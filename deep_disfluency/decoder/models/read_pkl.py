import pickle
import pandas as pd
swbd=pd.read_csv('swbd_DB_disf1_tags_table.csv')
# df1=pickle.load(open('DB_disf1_tags.pkl', "rb"))
# df1=pickle.load(open('swbd_DB_disf1_tags.pkl', "rb"))
# print(df1)
f = open("swbd_DB_disf1_tags.pkl",'wb')
pickle.dump(swbd,f)
f.close()
# df2=pickle.load(open('DB_disf1.pkl', "rb"))
# print(df2)
