# sklearn的学习
# LogisticRegression
from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # 精确性分数

a = [1, 2, 3, 4, 5]
breast_data = load_breast_cancer()
X = breast_data.data

y = breast_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=420)
# %%
l1 = []
l2 = []
l1test = []
l2test = []

# %%
for i in np.linspace(0.05, 1, 19):
    lrl1 = LR(penalty="l1", solver="liblinear", C=i, max_iter=1000)
    lrl2 = LR(penalty="l2", solver="liblinear", C=i, max_iter=1000)

    lrl1 = lrl1.fit(X_train, y_train)
    lrl2 = lrl2.fit(X_train, y_train)

    l1.append(accuracy_score(lrl1.predict(X_train), y_train))
    l1test.append(accuracy_score(lrl1.predict(X_test), y_test))

    l2.append(accuracy_score(lrl2.predict(X_train), y_train))
    l2test.append(accuracy_score(lrl2.predict(X_test), y_test))
# %%
print(l1, "l1")
print(l2, "l2")

print(l2test)
# %%
# 画图
graph = [l1, l2, l1test, l2test]
color = ["green", "black", "lightgreen", "gray"]
label = ["L1", "L2", "L1test", "L2test"]

# plt.figure(figsize=(6, 6))
for i in np.arange(len(graph)):
    plt.plot(np.linspace(0.05, 1.5, 19), graph[i], color[i], label=label[i])
plt.legend(loc=4)  # 图例的位置在哪里?4表示，右下角
plt.show()

# %%

from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel

# %%

LR_C = LR(solver="liblinear", C=0.9, random_state=420)  # 逻辑回归
cr_test = cross_val_score(LR_C, X, y, cv=10).mean()  # 交叉验证

# %%%
X_embedded = SelectFromModel(LR_C, norm_order=1).fit_transform(X, y)  # 专门给逻辑回归降维的东西

print(X_embedded.shape)  # (569, 9)
cr_test1 = cross_val_score(LR_C, X_embedded, y, cv=10).mean()  # 交叉验证看一下效果
# %%
