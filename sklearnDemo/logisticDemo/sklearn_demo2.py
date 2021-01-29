# 梯度下降

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


# 不同的梯度下降的步长在训练集和测试的预测表现
l2 = []
l2test = []
l2_big = []
for i in np.arange(1, 201, 10):
    lrl2 = LR(penalty="l2", solver="liblinear", C=0.9, max_iter=i) # penalty是正则化
    lrl2 = lrl2.fit(X_train, y_train)
    l2.append(accuracy_score(lrl2.predict(X_train), y_train))  # 训练集的评估分数
    l2test.append(accuracy_score(lrl2.predict(X_test), y_test))  # 测试集的评估分数



# %%
for i in np.arange(1, 201, 10):
    lrl2 = LR(penalty="l2", solver="liblinear", C=0.9, max_iter=500)
    lrl2 = lrl2.fit(X_train, y_train)
    # l2_big.append(accuracy_score(lrl2.predict(X_train), y_train))  # 训练集的评估分数
    l2_big.append(accuracy_score(lrl2.predict(X_test), y_test))  # 测试集的评估分数
# %%
graph = [l2, l2test,l2_big]
color = ["black", "gray","red"]
label = ["L2", "L2test","l2_big"]

plt.figure(figsize=(9, 5))
for i in range(len(graph)):
    plt.plot(np.arange(1, 201, 10), graph[i], color=color[i], label=label[i])
plt.legend(loc=4)
plt.xticks(np.arange(1, 201, 10))
plt.xlabel("step")
plt.ylabel("(accuracy_score")
plt.show()

# %%

