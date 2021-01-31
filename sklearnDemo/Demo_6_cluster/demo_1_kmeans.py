# %%

# 本文主要展示聚类算法
from sklearn.datasets import make_blobs
import numpy as  np
import matplotlib.pyplot as plt

# %%

# 自己创建数据集
X, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)

# %%

# 画图

# fig, ax1 = plt.subplots(1)
# ax1.scatter(X[:, 0], X[:, 1]#.scatter散点图
#             ,marker='o' #点的形状
#             ,s=8 #点的大小
#            )
# plt.show()
# %%

# 如果我们想要看见这个点的分布，怎么办？
color = ["red", "pink", "orange", "gray"]
fig, ax1 = plt.subplots(1)

for i in range(4):
    ax1.scatter(X[y == i, 0], X[y == i, 1]  # np 才有的boolean索引
                , marker='o'  # 点的形状
                , s=8  # 点的大小
                , c=color[i]
                )
plt.show()

# %%

from sklearn.cluster import KMeans

n_clusters = 4
# %%
cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

# %%
# 重要属性Labels_，查看聚好的类别，每个样本所对应的类
y_pred = cluster.labels_
print(y_pred)

# %%
# 如果我们想要看见这个点的分布，怎么办？
color = ["red", "pink", "orange", "gray"]

fig, ax1 = plt.subplots(1)
for i in range(4):
    ax1.scatter(X[y_pred == i, 0], X[y_pred == i, 1]  # np 才有的boolean索引
                , marker='o'  # 点的形状
                , s=8  # 点的大小
                , c=color[i]
                )
plt.show()
# %%

from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
# %%
silhouette_score(X,y_pred)