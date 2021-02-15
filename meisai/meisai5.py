# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def guiyi(df):
    df = (df - df.min()) / (df.max() - df.min())
    return df


# %%
influence_data = pd.read_csv('./influence_data.csv', encoding='gbk')  # Pop/Rock
# %%

# Pop/Rock          24141
# R&B;               5530
# Country            3301
# Jazz               2716
# Vocal
gener_name = 'Pop/Rock'
pop_man = influence_data[influence_data["influencer_main_genre"] == gener_name].loc[:,
          ['influencer_id', 'influencer_name', 'influencer_main_genre', 'influencer_active_start']]
pop_man = pop_man.drop_duplicates("influencer_name")

# %%
RB_man = influence_data[influence_data["influencer_main_genre"] == "R&B;"].loc[:,
         ['influencer_id', 'influencer_name', 'influencer_main_genre', 'influencer_active_start']]
RB_man = RB_man.drop_duplicates("influencer_name")
RB_man.index = np.arange(len(RB_man))
# %%
country_man = influence_data[influence_data["influencer_main_genre"] == "Country"].loc[:,
              ['influencer_id', 'influencer_name', 'influencer_main_genre', 'influencer_active_start']]
country_man = country_man.drop_duplicates("influencer_name")
country_man.index = np.arange(len(country_man))
# %%

# 歌曲样本数据集的去数组化
full_music_sample = pd.read_csv('./full_music_data_sample.csv')
full_music_sample["artist_names"] = full_music_sample["artist_names"].apply(eval)  # 字符串转换成数组


def get_x(x):
    return x[0]


full_music_sample["artist_names"] = full_music_sample["artist_names"].apply(get_x)

# %%
# full_music_sample[full_music_sample["artist_names"].isin(pop_man["influencer_name"])]

pop_rock_song = pd.read_csv('./pop_rock_song.csv')

# %%
# 成体系化
songs = full_music_sample[full_music_sample["artist_names"].isin(pop_man["influencer_name"])]
songs_feature = guiyi(songs.loc[:, ['danceability', 'energy', 'valence',
                                    'tempo', 'loudness', 'mode', 'key', 'acousticness', 'instrumentalness',
                                    'liveness', 'speechiness', 'explicit', 'duration_ms', 'popularity']])
songs_feature = songs_feature.median()
# %%
# =======自己设置开始============
# 标签
labels = songs_feature.index.tolist()
# 数据个数
dataLenth = songs_feature.shape[0]
# 数据
data = songs_feature.values.tolist()
# ========自己设置结束============

angles = np.linspace(0, 2 * np.pi, dataLenth, endpoint=False)
labels = np.concatenate((labels, [labels[0]]))  # 闭合
data = np.concatenate((data, [data[0]]))  # 闭合
angles = np.concatenate((angles, [angles[0]]))  # 闭合

fig = plt.figure()
ax = fig.add_subplot(111, polar=True)  # polar参数！！
ax.plot(angles, data, 'bo-', linewidth=2)  # 画线
ax.fill(angles, data, facecolor='r', alpha=0.25)  # 填充
ax.set_thetagrids(angles * 180 / np.pi, labels, fontproperties="SimHei")
ax.set_title(gener_name, va='bottom', fontproperties="SimHei")
ax.set_rlim(0, 1)
ax.grid(True)
# %%
country_song = full_music_sample[full_music_sample["artist_names"].isin(country_man["influencer_name"])]
RB_song = full_music_sample[full_music_sample["artist_names"].isin(RB_man["influencer_name"])]
# %%
country_song_feature = guiyi(country_song.loc[:, ['danceability', 'energy', 'valence',
                                                  'tempo', 'loudness', 'mode', 'key', 'acousticness',
                                                  'instrumentalness',
                                                  'liveness', 'speechiness', 'explicit', 'duration_ms', 'popularity']])
RB_song_feature = guiyi(RB_song.loc[:, ['danceability', 'energy', 'valence',
                                        'tempo', 'loudness', 'mode', 'key', 'acousticness', 'instrumentalness',
                                        'liveness', 'speechiness', 'explicit', 'duration_ms', 'popularity']])
# %%
country_song_feature = country_song_feature.median()
RB_song_feature = RB_song_feature.median()

# %%
# =======自己设置开始============
# 标签
labels = RB_song_feature.index.tolist()
# 数据个数
dataLenth = RB_song_feature.shape[0]
# 数据
data = RB_song_feature.values.tolist()
# ========自己设置结束============

angles = np.linspace(0, 2 * np.pi, dataLenth, endpoint=False)
labels = np.concatenate((labels, [labels[0]]))  # 闭合
data = np.concatenate((data, [data[0]]))  # 闭合
angles = np.concatenate((angles, [angles[0]]))  # 闭合

fig = plt.figure()
ax = fig.add_subplot(111, polar=True)  # polar参数！！
ax.plot(angles, data, 'bo-', linewidth=2)  # 画线
ax.fill(angles, data, facecolor='r', alpha=0.25)  # 填充
ax.set_thetagrids(angles * 180 / np.pi, labels, fontproperties="SimHei")
ax.set_title("RB_song_feature", va='bottom', fontproperties="SimHei")
ax.set_rlim(0, 1)
ax.grid(True)
# %%
pop_rock_song["release_date"] = pd.to_datetime(pop_rock_song["release_date"])
pop_rock_song = pop_rock_song.drop_duplicates("release_date")
# %%
pop_rock_song = pop_rock_song.set_index('release_date').sort_values("release_date")
# %%
pop_rock_song = pop_rock_song.drop("year", axis=1)
# %%
pop_rock_song_feature = pop_rock_song.loc[:, ['danceability', 'energy', 'valence',
                                              'tempo', 'loudness', 'key', 'acousticness', 'instrumentalness',
                                              'liveness', 'speechiness', 'duration_ms', 'popularity']]

cols = pop_rock_song_feature.columns
index1 = pop_rock_song_feature.index

# %%


pop_rock_song_feature.loc[:, ["tempo", "key", "loudness", "duration_ms", "popularity"]] = guiyi(
    pop_rock_song_feature.loc[:, ["tempo", "key", "loudness", "duration_ms", "popularity"]])
# %%
pop_rock_song_feature = pop_rock_song_feature.drop(
    ["liveness", "speechiness", "duration_ms", "instrumentalness", "key", "acousticness", "loudness"], axis=1)

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit_transform([pop_rock_song_feature.loc[:, "duration_ms"]])

scaler.fit_transform(pop_rock_song_feature.to_numpy())
# %%
# pop_rock_song_feature.loc[:,["tempo","key",	"loudness","duration_ms","popularity"]] = guiyi(
#     pop_rock_song_feature.loc[:,["tempo","key",	"loudness","duration_ms","popularity"]])

pop_rock_song_feature = pd.DataFrame(scaler.fit_transform(pop_rock_song_feature.to_numpy()),
                                     columns=cols,
                                     index=index1)
# %%

from matplotlib.lines import Line2D
from matplotlib import rcParams, cycler

cmap = plt.cm.coolwarm
rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, pop_rock_song_feature.shape[1])))
# %%
fig, ax = plt.subplots()
lines = ax.plot(pop_rock_song_feature)
ax.legend(pop_rock_song_feature.columns)
# %%
N = 10
data = (np.geomspace(1, 10, 100) + np.random.randn(N, 100)).T


# %%
def entropy(data0):
    # 返回每个样本的指数
    # 样本数，指标个数
    n, m = np.shape(data0)
    # 一行一个样本，一列一个指标
    # 下面是归一化
    maxium = np.max(data0, axis=0)
    minium = np.min(data0, axis=0)
    data = (data0 - minium) * 1.0 / (maxium - minium)
    ##计算第j项指标，第i个样本占该指标的比重
    sumzb = np.sum(data, axis=0)
    data = data / sumzb
    # 对ln0处理
    a = data * 1.0
    a[np.where(data == 0)] = 0.0001
    #    #计算每个指标的熵
    e = (-1.0 / np.log(n)) * np.sum(data * np.log(a), axis=0)
    #    #计算权重
    w = (1 - e) / np.sum(1 - e)
    recodes = np.sum(data0 * w, axis=1)
    return recodes


ennn = entropy(pop_rock_song_feature.to_numpy().T)

# %%
pop_rock_song_feature = pop_rock_song_feature.reset_index()
pop_rock_song_feature["release_date"]=pop_rock_song_feature["release_date"].dt.year

# %%
def get_data(x) :
    return x.mean()
pop_rock_song_feature = pop_rock_song_feature.groupby("release_date").apply(get_data)
# %%
pop_rock_song_feature.drop("release_date",index=1)
# %%
from statsmodels.tsa.arima_model import ARIMA
# pop_rock_song_feature[pop_rock_song_feature.index >= "1956-10-19"] # 大于某一个日期
import seaborn as sns

fig, axes = plt.subplots(2, 2)  # fig是整个画布，axes是子图,1，2表示1行两列
sns.lineplot(data=pop_rock_song_feature.iloc[:, 0:2], palette="tab10", linewidth=1.5, ax=axes[0][0])
# plt.subplots_adjust(wspace=0.5)  # 子图很有可能左右靠的很近，调整一下左右距离
sns.lineplot(data=pop_rock_song_feature.iloc[:, 2:4], palette="tab10", linewidth=1.5, ax=axes[0][1])
sns.lineplot(data=pop_rock_song_feature.iloc[:, 4:6], palette="tab10", linewidth=1.5, ax=axes[1][0])
sns.lineplot(data=pop_rock_song_feature.iloc[:, 6:8], palette="tab10", linewidth=1.5, ax=axes[1][1])
fig.set_figwidth(40)  # 这个是设置整个图（画布）的大小
