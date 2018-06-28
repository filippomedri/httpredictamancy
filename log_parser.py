
import pandas as pd
import os

df = pd.read_csv(os.path.join('resources','UofS_access_log'),sep='\s+',header=None, engine='python',error_bad_lines=False)
pd.to_datetime(df[3], format="[%d/%b/%Y:%H:%M:%S")

new_df =  df[[0,3,4,5,7,8]]
new_df.rename(columns = {0:'client',3:'time',5:'command',6:'file',7:'return_code',8:'bytes'})
ts1 = df[3]
ts2 = df[[3,8]]
new_df.to_csv('cleaned_df')
ts1.to_csv('ts1')
ts2.to_csv('ts2')
