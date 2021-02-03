# %%

# 本文主要讨论相关性的操作
import random
import numpy as np
# a = [random.randint(0, 100) for a in range(20)]
a = np.random.randint(1,100,20)
# b = [random.randint(0, 100) for a in range(20)]
b = np.random.randint(1,100,20)

#%%

# 先构造一个矩阵
ab = np.array([a, b])
# 计算协方差矩阵
ab_cov = np.cov(ab)

# %%
#相关性系数
ab_corr = np.corrcoef(ab)

# %%
import pandas as pd
# 使用 DataFrame 作为数据结构，为方便计算，我们会将 ab 矩阵转置
dfab = pd.DataFrame(ab.T, columns=['A', 'B'])
# # A B 协方差
dfAcovB = dfab.A.cov(dfab.B)
# # A B 相关系数
dfAcorrB = dfab.A.corr(dfab.B)
# dfab