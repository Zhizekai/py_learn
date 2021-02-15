# %%
import numpy as np
import pandas as pd

# %%

full_music_data = pd.read_csv('./full_music_data.csv', chunksize=20000)
# %%
full_music_data_sample = pd.DataFrame([])
for chunk in full_music_data:
    full_music_data_sample = pd.concat([full_music_data_sample, chunk.sample(5000)], axis=0)
    print(chunk.shape)
# %%

full_music_data_sample.index = np.arange(len(full_music_data_sample))
# %%
full_music_data_sample = full_music_data_sample.drop('Unnamed: 19', axis=1)
# %%


# full_music_data_sample.groupby("artist_names").apply()

music = full_music_data_sample["artist_names"].value_counts()
music = music[music > 20]
music = pd.DataFrame({"artist_names": music.index, "song_val": music.values})
music.to_csv("./music.csv")

# %%
full_music_data_sample = full_music_data_sample.sort_values("popularity", ascending=False)
full_music_data_sample.to_csv("./full_music_data_sample.csv", index=False)
# %%
full_corr = full_music_data_sample.corr(method='pearson')
full_corr.to_csv("./full_corr.csv")
