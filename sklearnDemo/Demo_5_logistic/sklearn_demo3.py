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
X, y = sm.fit_sample(X, y)  # 样本均衡化之后的X和y

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
# %%
model_data["qcut"], updown = pd.qcut(model_data["age"], retbins=True, q=20)  # 等频分箱

"""
pd.qcut，基于分位数的分箱函数，本质是将连续型变量离散化
只能够处理一维数据。返回箱子的上限和下限
参数q：要分箱的个数
参数retbins=True来要求同时返回结构为索引为样本索引，元素为分到的箱子的Series
现在返回两个值：每个样本属于哪个箱子，以及所有箱子的上限和下限
"""
# 在这里时让model_data新添加一列叫做“分箱”，这一列其实就是每个样本所对应的箱子
# print(model_data.head())
# print(model_data["qcut"])
# print(model_data["qcut"].value_counts())

# 所有箱子的上限和下限
# print(updown)

# %%

# 统计每个分箱中0和1的数量
# 这里使用了数据透视表的功能groupby
coount_y0 = model_data[model_data["SeriousDlqin2yrs"] == 0].groupby(by="qcut").count()["SeriousDlqin2yrs"]

coount_y1 = model_data[model_data["SeriousDlqin2yrs"] == 1].groupby(by="qcut").count()["SeriousDlqin2yrs"]

# num_bins值分别为每个区间的上界，下界，0出现的次数，1出现的次数
num_bins = [*zip(updown, updown[1:], coount_y0, coount_y1)]

# 注意zip会按照最短列来进行结合
print(num_bins)
# %%

for i in range(20):
    #如果第一个组没有包含正样本或负样本，向后合并
    if 0 in num_bins[0][2:]:
        num_bins[0:2] = [(
            num_bins[0][0],
            num_bins[1][1],
            num_bins[0][2]+num_bins[1][2],
            num_bins[0][3]+num_bins[1][3])]
        continue

    """
    合并了之后，第一行的组是否一定有两种样本了呢？不一定
    如果原本的第一组和第二组都没有包含正样本，或者都没有包含负样本，那即便合并之后，第一行的组也还是没有
    包含两种样本
    所以我们在每次合并完毕之后，还需要再检查，第一组是否已经包含了两种样本
    这里使用continue跳出了本次循环，开始下一次循环，所以回到了最开始的for i in range(20), 让i+1
    这就跳过了下面的代码，又从头开始检查，第一组是否包含了两种样本
    如果第一组中依然没有包含两种样本，则if通过，继续合并，每合并一次就会循环检查一次，最多合并20次
    如果第一组中已经包含两种样本，则if不通过，就开始执行下面的代码
    """
    #已经确认第一组中肯定包含两种样本了，如果其他组没有包含两种样本，就向前合并
    #此时的num_bins已经被上面的代码处理过，可能被合并过，也可能没有被合并
    #但无论如何，我们要在num_bins中遍历，所以写成in range(len(num_bins))
    for i in range(len(num_bins)):
        if 0 in num_bins[i][2:]:
            num_bins[i-1:i+1] = [(
                num_bins[i-1][0],
                num_bins[i][1],
                num_bins[i-1][2]+num_bins[i][2],
                num_bins[i-1][3]+num_bins[i][3])]
        break
        #如果对第一组和对后面所有组的判断中，都没有进入if去合并，则提前结束所有的循环
    else:
        break

    """
    这个break，只有在if被满足的条件下才会被触发
    也就是说，只有发生了合并，才会打断for i in range(len(num_bins))这个循环
    为什么要打断这个循环？因为我们是在range(len(num_bins))中遍历
    但合并发生后，len(num_bins)发生了改变，但循环却不会重新开始
    举个例子，本来num_bins是5组，for i in range(len(num_bins))在第一次运行的时候就等于for i in 
    range(5)
    range中输入的变量会被转换为数字，不会跟着num_bins的变化而变化，所以i会永远在[0,1,2,3,4]中遍历
    进行合并后，num_bins变成了4组，已经不存在=4的索引了，但i却依然会取到4，循环就会报错
    因此在这里，一旦if被触发，即一旦合并发生，我们就让循环被破坏，使用break跳出当前循环
    循环就会回到最开始的for i in range(20)中
    此时判断第一组是否有两种标签的代码不会被触发，但for i in range(len(num_bins))却会被重新运行
    这样就更新了i的取值，循环就不会报错了
    """


# %%

# 计算WOE和BAD RATE
# BAD RATE与bad%不是一个东西
# BAD RATE是一个箱中，坏的样本所占的比例 (bad/total)
# 而bad%是一个箱中的坏样本占整个特征中的坏样本的比例

def get_woe(num_bins):
    # 通过 num_bins 数据计算 woe
    columns = ["min", "max", "count_0", "count_1"]
    df = pd.DataFrame(num_bins, columns=columns)

    df["total"] = df.count_0 + df.count_1  # 一个箱子当中所有的样本数
    df["percentage"] = df.total / df.total.sum()  # 一个箱子里的样本数，占所有样本的比例
    df["bad_rate"] = df.count_1 / df.total  # 一个箱子坏样本的数量占一个箱子里边所有样本数的比例
    df["good%"] = df.count_0 / df.count_0.sum()
    df["bad%"] = df.count_1 / df.count_1.sum()
    df["woe"] = np.log(df["good%"] / df["bad%"])
    return df


# 计算IV值
def get_iv(df):
    rate = df["good%"] - df["bad%"]
    iv = np.sum(rate * df.woe)
    return iv
# %%

num_bins_ = num_bins.copy()

import matplotlib.pyplot as plt
import scipy

IV = []
axisx = []

while len(num_bins_) > 2:  # 大于设置的最低分箱个数
    pvs = []
    # 获取 num_bins_两两之间的卡方检验的置信度（或卡方值）
    for i in range(len(num_bins_) - 1):
        x1 = num_bins_[i][2:]
        x2 = num_bins_[i + 1][2:]
        # 0 返回 chi2 值，1 返回 p 值。
        pv = scipy.stats.chi2_contingency([x1, x2])[1]  # p值
        # chi2 = scipy.stats.chi2_contingency([x1,x2])[0]#计算卡方值
        pvs.append(pv)

    # 通过 p 值进行处理。合并 p 值最大的两组
    i = pvs.index(max(pvs))
    num_bins_[i:i + 2] = [(
        num_bins_[i][0],
        num_bins_[i + 1][1],
        num_bins_[i][2] + num_bins_[i + 1][2],
        num_bins_[i][3] + num_bins_[i + 1][3])]

    bins_df = get_woe(num_bins_)
    axisx.append(len(num_bins_))
    IV.append(get_iv(bins_df))

plt.figure()
plt.plot(axisx, IV)
plt.xticks(axisx)
plt.xlabel("number of box")
plt.ylabel("IV")
plt.show()
# 选择转折点处，也就是下坠最快的折线点，所以这里对于age来说选择箱数为6