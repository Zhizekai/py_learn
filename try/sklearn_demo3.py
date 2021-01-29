# %%
# 评分卡模型案例
import numpy as np
import pandas as pd

# %%
# 导入数据
rankingCard_data = pd.read_csv(r".\rankingcard.csv", index_col=0)

# %%
# 设置显示的最大列、宽等参数，消掉打印不完全中间的省略号
# pd.set_option('display.max_columns', 1000)
# pd.set_option('display.width', 1000)  # 加了这一行那表格的一行就不会分段出现了
# pd.set_option('display.max_colwidth', 1000)
# pd.set_option('display.height', 1000)
# 显示所有列
# pd.set_option('display.max_columns', None)
# 显示所有行
# pd.set_option('display.max_rows', None)
# %%
print(rankingCard_data.head())

print(rankingCard_data.shape)
print(rankingCard_data.info())  # 数据信息
# %%

rankingCard_data.duplicated().sum()  # 查看有多少行重复的
rankingCard_data = rankingCard_data.drop_duplicates()

rankingCard_data.info()
rankingCard_data.index = range(rankingCard_data.shape[0])
rankingCard_data.info()
# %%

# 探索缺失
print(rankingCard_data.isnull().sum(), "\n")  # 这个sum是每一列相加
print(rankingCard_data.isnull().sum() / rankingCard_data.shape[0])  # 缺失的占每一列的比例
# %%

# #这里用均值填补家庭人数这一项
rankingCard_data["NumberOfDependents"].fillna(rankingCard_data["NumberOfDependents"].mean(), inplace=True)
rankingCard_data["NumberOfDependents"].isnull().sum()
print(rankingCard_data.isnull().sum() / rankingCard_data.shape[0])

# %%
from sklearn.ensemble import RandomForestRegressor as rfr_

# 选出有缺失值的那一行的数据
print(rankingCard_data["MonthlyIncome"][rankingCard_data["MonthlyIncome"].isnull()])


def fill_missing_rf(X, y, to_fill, rfr):
    """
       使用随机森林填补一个特征的缺失值的函数

       参数：
       X：要填补的特征矩阵
       y：完整的，没有缺失值的标签
       to_fill：字符串，要填补的那一列的名称
       """
    # 构建我们的新特征矩阵和新标签
    df = X.copy()  # 复制X
    fill = df.loc[:, to_fill]  # 要填补的那一列数据
    df = pd.concat([df.loc[:, df.columns != to_fill], pd.DataFrame(y)], axis=1)  # 拼接数据

    # 找出我们的训练集和测试集
    Ytrain = fill[fill.notnull()]
    Ytest = fill[fill.isnull()]
    Xtrain = df.iloc[Ytrain.index, :]
    Xtest = df.iloc[Ytest.index, :]
    # 用随机森林回归来填补缺失值
    rfr = rfr(n_estimators=100)
    rfr = rfr.fit(Xtrain, Ytrain)
    Ypredict = rfr.predict(Xtest)

    return Ypredict


# %%
X_fill_null = rankingCard_data.iloc[:, 1:]  # 除了第一列的所有数据
y_fill_null = rankingCard_data["SeriousDlqin2yrs"]  # 第一列的数据
print(X_fill_null.shape)

y_pred = fill_missing_rf(X_fill_null, y_fill_null, "MonthlyIncome", rfr_)

# 查看y_pred是否和缺失的数据数量一致
print(y_pred == rankingCard_data.loc[:, "MonthlyIncome"].isnull().sum())

# %%
# 覆盖数据
rankingCard_data.loc[rankingCard_data.loc[:, "MonthlyIncome"].isnull(), "MonthlyIncome"] = y_pred

rankingCard_data.info()

# %%
rankingCard_data_describe = rankingCard_data.describe([0.01, 0.1, 0.25, .5, .75, .9, .99]).T  # 平均分多少段

# %%
# 异常值也被我们观察到，年龄的最小值居然有0，这不符合银行的业务需求，即便是儿童账户也要至少8岁，我们可以
# 查看一下年龄为0的人有多少
(rankingCard_data["age"] == 0).sum()
# 发现只有一个人年龄为0，可以判断这肯定是录入失误造成的，可以当成是缺失值来处理，直接删除掉这个样本
rankingCard_data = rankingCard_data[rankingCard_data["age"] != 0]
# %%
# 删除奇怪的统计值
print(rankingCard_data[rankingCard_data.loc[:, "NumberOfTimes90DaysLate"] > 90])
print(rankingCard_data[rankingCard_data.loc[:, "NumberOfTimes90DaysLate"] > 90].count())
rankingCard_data.loc[:, "NumberOfTimes90DaysLate"].value_counts()  # 数值统计
rankingCard_data = rankingCard_data[rankingCard_data.loc[:, "NumberOfTimes90DaysLate"] < 90]
# 一定要恢复索引
rankingCard_data.index = range(rankingCard_data.shape[0])
rankingCard_data.info()

# %%
# 探索标签的分布
X = rankingCard_data.iloc[:, 1:]
y = rankingCard_data.iloc[:, 0]

y.value_counts()  # 查看每一类别值得数据量，查看样本是否均衡

n_sample = X.shape[0]

n_1_sample = y.value_counts()[1]
n_0_sample = y.value_counts()[0]

print('样本个数：{}; 1占{:.2%}; 0占{:.2%}'.format(n_sample, n_1_sample / n_sample, n_0_sample / n_sample))
# 样本个数：149165; 1占6.62%; 0占93.38%

# %%
# 解决样本不均衡问题

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)  # 实例化
X, y = sm.fit_sample(X, y) # 样本均衡化之后的X和y

n_sample_ = X.shape[0]  # 278584

pd.Series(y).value_counts()

n_1_sample = pd.Series(y).value_counts()[1]
n_0_sample = pd.Series(y).value_counts()[0]

print('样本个数：{}; 1占{:.2%}; 0占{:.2%}'.format(n_sample_, n_1_sample / n_sample_, n_0_sample / n_sample_))
# 样本个数：278584; 1占50.00%; 0占50.00%

# %%

# 构建训练数据和测试数据


from sklearn.model_selection import train_test_split

X = pd.DataFrame(X)
y = pd.DataFrame(y)

X_train, X_vali, Y_train, Y_vali = train_test_split(X, y, test_size=0.3, random_state=420)
model_data = pd.concat([Y_train, X_train], axis=1)  # 训练数据构建模型
model_data.index = range(model_data.shape[0])
model_data.columns = rankingCard_data.columns

vali_data = pd.concat([Y_vali, X_vali], axis=1)  # 验证集
vali_data.index = range(vali_data.shape[0])
vali_data.columns = rankingCard_data.columns

model_data.to_csv(r".\model_data.csv")  # 训练数据
vali_data.to_csv(r".\vali_data.csv")  # 验证数据
