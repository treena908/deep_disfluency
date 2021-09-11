import pandas as pd
df1=pd.read_csv('data_w_pos.csv')
df2=pd.read_csv('data_w_pos_refined.csv')

print('len %d %d'%(len(df1),len(df2)))