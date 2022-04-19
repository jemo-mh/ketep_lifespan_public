#%%
from time import time
import pandas as pd
import numpy as np


# def Excel2txt():
df = pd.read_excel('/mnt/e/Workspace/jm/Projects/baby_eval/Dataset/Integrated_val.xlsx', sheet_name='Sheet2', usecols=[0,1,2,3,8,9,10,11,12,13,14,15,17], skiprows=0, nrows=1000, index_col= False)
print(df)

#%%
a =df.iloc[:,[4,5,6,7]]
a_np = a.to_numpy()
print(a_np)

#%%
# print(a_np.shape[0])
munjin=[]
for i in range(a_np.shape[0]):
    row = a_np[i,:]
    mj = np.sum(row)
    # print(mj)
    # mj = round(mj*2.5)

    munjin.append(mj)
munjin = pd.DataFrame(munjin)

# b= df.iloc[:,[0,1,2,3,8,9,10,12,13,15,17]]
b= df.iloc[:,[0,1,2,3,8,9,11,12]]
print(b)
#%%
b.insert(4,'munjin',munjin)
b.rename(columns=b.iloc[0], inplace = True)
b.drop([0], inplace=True)
print(b)
#%%
b.to_csv("Lifespan_validation0414.txt", index= False,  sep=',')
# # %%
# # def LifeSpan_Excel2txt():
# df = pd.read_excel('/mnt/e/Workspace/jm/Projects/ketep_lifespan/HI/Dataset/exel/LifeSpan_val.xlsx', skiprows=2, usecols=[1,2,3,5], nrows=99)
# df.to_csv("LifeSpan_Val.txt", index=False, sep=',')
# print(df)
# LifeSpan_Excel2txt()
# %%
