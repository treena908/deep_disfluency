import pickle
# df1=pickle.load(open('DB_disf1_tags.pkl', "rb"))
df1=pickle.load(open('swbd_disf1_tags.pkl', "rb"))
print(df1)
f = open("DB_disf1.pkl",'wb')
pickle.dump(df1,f)
f.close()
# df2=pickle.load(open('DB_disf1.pkl', "rb"))
# print(df2)
