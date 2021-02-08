# %%
import numpy as np
import pandas as pd

# %%
# 同一流派的音乐家
full_music_data_sample = pd.read_csv("./full_music_data_sample.csv")
# %%
influence_man_data = pd.read_csv('./influence_man_data3.csv')

# %%
pop_rock_influence_man = influence_man_data[influence_man_data["influencer_main_genre"] == "Pop/Rock"]
full_music_data_sample["artist_names"] = full_music_data_sample["artist_names"].apply(eval)  # 字符串转换成数组


# %%

def get_x(x):
    print(x[0])
    return x[0]


full_music_data_sample["artist_names"] = full_music_data_sample["artist_names"].apply(get_x)
pop_rock_song = full_music_data_sample[
    full_music_data_sample["artist_names"].isin(pop_rock_influence_man["influencer_name"])]
# %%
pop_rock_song.index = np.arange(len(pop_rock_song))
pop_rock_song.to_csv('./pop_rock_song.csv', index=False)
# %%
pop_rock_song.corr().to_csv('./pop_rock_song_corr.csv')
# %%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pop_rock_song = pd.read_csv('./pop_rock_song.csv')
pca_line = PCA().fit(pop_rock_song.loc[:1000,
                     ["danceability", "energy", "valence", "tempo", "acousticness", "liveness", "speechiness",
                      "popularity"]])


# %%
def plt_plant():
    plt.plot(np.arange(1, len(pca_line.explained_variance_ratio_) + 1), pca_line.explained_variance_ratio_)
    plt.xticks(np.arange(1, len(pca_line.explained_variance_ratio_) + 1))  # 这是为了限制坐标轴显示为整数
    plt.xlabel("number of components after dimension reduction")
    plt.ylabel("cumulative explained variance ratio")
    plt.show()


# %%
pca_dir = PCA(2).fit_transform(pop_rock_song.loc[:1000,
                               ["danceability", "energy", "valence", "tempo", "acousticness", "liveness", "speechiness",
                                "popularity"]])


# %%
def plant_julei():
    plt.scatter(pca_dir[:, 0], pca_dir[:, 1], marker='x', color='red', s=40)
    plt.show()


# %%
import seaborn as sns

pop_rock_song_corr = pd.read_csv('./pop_rock_song_corr.csv')
# %%
pop_sam = pop_rock_song.loc[:1000,
          ["danceability", "energy", "valence", "tempo", "acousticness", "liveness", "speechiness", "popularity"]]


# %%

def sns_plot():
    sns.pairplot(pop_sam.corr(), hue="danceability")
    sns.clustermap(pop_sam.corr())
    sns.jointplot(x=pca_dir[:, 0], y=pca_dir[:, 1], data=pca_dir, kind='kde')


# %%
# def get_data_by_year():
data_by_year = pd.read_csv('./data_by_year.csv')
data_by_year["year"] = pd.to_datetime(data_by_year["year"], format="%Y")

data_by_year = data_by_year.set_index('year')


# %%
def guiyi(df):
    df = (df - df.min()) / (df.max() - df.min())
    return df


data_by_year["duration_ms"] = guiyi(data_by_year["duration_ms"])
data_by_year["tempo"] = guiyi(data_by_year["tempo"])
data_by_year["loudness"] = guiyi(data_by_year["loudness"])
data_by_year["key"] = guiyi(data_by_year["key"])
data_by_year["popularity"] = guiyi(data_by_year["popularity"])
# get_data_by_year()

# %%

sns.lineplot(data=data_by_year.drop(["loudness", "key", "mode"], axis=1), palette="tab10", linewidth=2.5)
# %%
full_sample = pd.read_csv('./full_music_data_sample.csv')
full_sample["artist_names"] = full_sample["artist_names"].apply(eval)  # 字符串转换成数组


def get_x(x):
    print(x[0])
    return x[0]


full_sample["artist_names"] = full_sample["artist_names"].apply(get_x)
# %%
pop_rock_song = pd.read_csv('./pop_rock_song.csv')
# %%
pop_rock_song["artist_names"].value_counts().to_csv('./song_nums.csv')

# %%
# The Kinks   The Beatles
the_kinks_songs = pop_rock_song[pop_rock_song["artist_names"] == "The Kinks"]
the_kinks_songs.index = np.arange(len(the_kinks_songs))
the_kinks_songs = guiyi(the_kinks_songs.loc[:, ['danceability', 'energy', 'valence',
                                                'tempo', 'loudness', 'mode', 'key', 'acousticness', 'instrumentalness',
                                                'liveness', 'speechiness', 'duration_ms', 'popularity']])

the_beatles_songs = pop_rock_song[pop_rock_song["artist_names"] == "The Beatles"]
the_beatles_songs.index = np.arange(len(the_beatles_songs))
the_beatles_songs = guiyi(the_beatles_songs.loc[:, ['danceability', 'energy', 'valence',
                                                    'tempo', 'loudness', 'mode', 'key', 'acousticness',
                                                    'instrumentalness',
                                                    'liveness', 'speechiness', 'duration_ms', 'popularity']])

# %%
the_kinks = pd.DataFrame({"tezheng": the_kinks_songs.median().index, "value": the_kinks_songs.median().values})

the_beatles = pd.DataFrame({"tezheng": the_beatles_songs.median().index, "value": the_beatles_songs.median().values})

# %%
# =======自己设置开始============
# 标签
labels = the_kinks["tezheng"].to_list()
# 数据个数
dataLenth = the_kinks["tezheng"].shape[0]
# 数据
data = the_kinks["value"].to_list()
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
ax.set_title("the_kinks", va='bottom', fontproperties="SimHei")
ax.set_rlim(0, 1)
ax.grid(True)
