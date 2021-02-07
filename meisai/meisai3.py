# %%
import numpy as np
import pandas as pd

# %%
# 同一流派的音乐家
# full_music_data_sample = pd.read_csv("./full_music_data_sample.csv")
# # %%
# influence_man_data = pd.read_csv('./influence_man_data3.csv')
#
# # %%
# pop_rock_influence_man = influence_man_data[influence_man_data["influencer_main_genre"] == "Pop/Rock"]
# full_music_data_sample["artist_names"] = full_music_data_sample["artist_names"].apply(eval)  # 字符串转换成数组
#
#
# # %%
#
# def get_x(x):
#     print(x[0])
#     return x[0]
#
#
# full_music_data_sample["artist_names"] = full_music_data_sample["artist_names"].apply(get_x)
# pop_rock_song = full_music_data_sample[
#     full_music_data_sample["artist_names"].isin(pop_rock_influence_man["influencer_name"])]
# # %%
# pop_rock_song.index = np.arange(len(pop_rock_song))
# pop_rock_song.to_csv('./pop_rock_song.csv',index=False)
# # %%
# pop_rock_song.corr().to_csv('./pop_rock_song_corr.csv')
# %%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
pop_rock_song = pd.read_csv('./pop_rock_song.csv')
pca_line = PCA().fit(pop_rock_song.loc[:1000,["danceability","energy","valence","tempo", "acousticness","liveness","speechiness","popularity"]])

# %%
plt.plot(np.arange(1,len(pca_line.explained_variance_ratio_)+1),pca_line.explained_variance_ratio_)
plt.xticks(np.arange(1,len(pca_line.explained_variance_ratio_)+1)) #这是为了限制坐标轴显示为整数
plt.xlabel("number of components after dimension reduction")
plt.ylabel("cumulative explained variance ratio")
plt.show()
# %%
pca_dir = PCA(2).fit_transform(pop_rock_song.loc[:1000,["danceability","energy","valence","tempo", "acousticness","liveness","speechiness","popularity"]])
# %%
plt.scatter(pca_dir[:,0], pca_dir[:,1], marker = 'x',color = 'red', s = 40 )
plt.show()
# %%
import seaborn as sns
pop_rock_song_corr = pd.read_csv('./pop_rock_song_corr.csv')
# %%
pop_sam = pop_rock_song.loc[:1000,["danceability","energy","valence","tempo", "acousticness","liveness","speechiness","popularity"]]
# %%
sns.pairplot(pop_sam.corr(),hue="danceability")
# %%
sns.clustermap(pop_sam.corr())
# %%
sns.jointplot(x = pca_dir[:,0], y = pca_dir[:,1], data = pca_dir ,kind = 'kde')