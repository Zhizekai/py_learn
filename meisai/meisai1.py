# %%
import numpy as np
import pandas as pd

# %%
influence_data = pd.read_csv('./influence_data.csv', encoding='gbk')
data_by_year = pd.read_csv('./data_by_year.csv')
influence_man = influence_data["influencer_name"].value_counts()  # 统计影响者出现的次数

# %%
# 删除影响力小于20的影响者
influence_man = influence_man[influence_man > 20]
influence_man = pd.DataFrame({'person': influence_man.index, 'influence_value': influence_man.values})
# %%
influence_man.to_csv('./influence_man.csv', index=False)  # index = 0 :不保存行索引 column = 0: 不保存列索引
# %%
# 获取主要影响者全部的数据并且保存到csv
influence_man_data = influence_data[influence_data['influencer_name'].isin(influence_man["person"])]
influence_man_data = influence_man_data.sample(1000)  # 随机抽样 保证样本比例不变，降低数据量
influence_man_data.index = np.arange(influence_man_data.shape[0])
# influence_man_data.to_csv('./influence_man_full_data.csv', index=0)
print(influence_man_data)  # 主要影响者的全部数据

# %%
# 构建全部歌手的数据 并且去重
# 把follower和influencer两列的数据都拿出来，去重，axis=0方向拼接，就是垂直拼接
# influence_name_data = pd.DataFrame(influence_man_data["influencer_name"].drop_duplicates())  # 先取列 再去重
# influence_name_data = influence_man_data.loc[:, ["influencer_name", "influencer_main_genre"]].drop_duplicates(
#     ["influencer_name"])
# influence_name_data.columns = ["person", "main_genre"]
# follower_name_data = influence_man_data.loc[:, ["follower_name", "follower_main_genre"]].drop_duplicates(
#     ["follower_name"])
# follower_name_data.columns = ["person", "main_genre"]
influence_name_data = pd.DataFrame(influence_man_data["influencer_name"].drop_duplicates())
influence_name_data.columns = ["person"]
follower_name_data = pd.DataFrame(influence_man_data["follower_name"].drop_duplicates())
follower_name_data.columns = ["person"]

influence_man_nodes = pd.concat([influence_name_data, follower_name_data], axis=0)
influence_man_nodes = influence_man_nodes.drop_duplicates()
influence_man_nodes.index = np.arange(influence_man_nodes.shape[0])
print(influence_man_nodes)  # 所有歌手

# %%
# the_beatles = influence_man_data[influence_man_data["influencer_name"] == "The Beatles"]
# the_beatles.index = np.arange(the_beatles.shape[0])
#
# the_beatles_links = the_beatles.loc[:, ["influencer_name", "follower_name"]].loc[:100, :]["influencer_name"].to_list()
# others_links = the_beatles.loc[:, ["influencer_name", "follower_name"]].loc[:100, :]["follower_name"].to_list()
# links_data = []
# node_data = [{"name": "The Beatles"}]
# for i in np.arange(len(the_beatles_links)):
#     links_data.append({"source": others_links[i], "target": the_beatles_links[i], "value": 1})
#     node_data.append({"name": others_links[i]})

# %%
from pyecharts import options as opts
from pyecharts.charts import Graph

# %%
# nodes_data = [
#     opts.GraphNode(name="结点1", symbol_size=10),
#     opts.GraphNode(name="结点2", symbol_size=20),
#     opts.GraphNode(name="结点3", symbol_size=30),
#     opts.GraphNode(name="结点4", symbol_size=40),
#     opts.GraphNode(name="结点5", symbol_size=50),
#     opts.GraphNode(name="结点6", symbol_size=60),
# ]
#
# links_data = [
#     opts.GraphLink(source="结点1", target="结点2", value=2),
#     opts.GraphLink(source="结点2", target="结点3", value=3),
#     opts.GraphLink(source="结点3", target="结点4", value=4),
#     opts.GraphLink(source="结点4", target="结点5", value=5),
#     opts.GraphLink(source="结点5", target="结点6", value=6),
#     opts.GraphLink(source="结点6", target="结点1", value=7),
#     opts.GraphLink(source="结点1", target="结点6", value=8),
# ]
# %%
nodes_data = []
# 节点数据
for i in np.arange(len(influence_man_nodes)):
    nodes_data.append(opts.GraphNode(name=influence_man_nodes["person"][i], symbol_size=10, category=1))

# %%
links_data = []
for i in np.arange(len(influence_man_data)):
    links_data.append(
        opts.GraphLink(source=influence_man_data["follower_name"][i],
                       target=influence_man_data["influencer_name"][i], value=2))

# %%

c = (
    Graph(
        init_opts=opts.InitOpts(width="100%",  # 图宽
                                height="700px",  # 图高
                                renderer="canvas",  # 渲染模式 svg 或 canvas，即 RenderType.CANVAS 或 RenderType.SVG
                                )

    ).add(
        "",
        nodes_data,
        links_data,
        repulsion=50,
        linestyle_opts=opts.LineStyleOpts(curve=0.2),
        label_opts=opts.LabelOpts(is_show=False),
    ).set_global_opts(
        legend_opts=opts.LegendOpts(is_show=False),
        title_opts=opts.TitleOpts(title="Graph-GraphNode-GraphLink-WithEdgeLabel"),

    ).render("graph_with_edge_options.html")
)
# %%

# influence_man_data
influence_man.columns = ["influencer_name", "influence_value"]
influence_man_full_data = pd.merge(influence_man, influence_man_data, on=['influencer_name'])
influence_man_full_data = influence_man_full_data.drop_duplicates("influencer_name")
influence_man_full_data.index = np.arange(len(influence_man_full_data))
influence_man_full_data.to_csv('./influence_man_full_data.csv', index=False)


# %%

def get_influence_man(x):
    # x[x["follower_main_genre"].isin(x["influencer_main_genre"])]

    if x["influencer_name"].count() > 200:
        print(x["influencer_name"].count())
        print(x)

    x["is_in_person"] = x[x["follower_main_genre"].isin(x["influencer_main_genre"])]["follower_main_genre"].count()
    x["not_in_person"] = x[~x["follower_main_genre"].isin(x["influencer_main_genre"])]["follower_main_genre"].count()
    x["influence_val"] = x["follower_main_genre"].count()
    return x


influence_man_data2 = influence_man_data.groupby("influencer_name", as_index=False).apply(get_influence_man)
# influence_man_data2.columns = ["influencer_name","is_in_person"]
# %%
influence_man_data2 = influence_man_data2.sort_values("influence_val", ascending=False)
influence_man_data2 = influence_man_data2.drop_duplicates("influencer_name")
influence_man_data2.index = np.arange(len(influence_man_data2))
influence_man_data2.to_csv("./influence_man_data2.csv", index=False)
# pd.concat(influence_man_data2[])
# %%
influence_man_data2["is_in_person"] = (influence_man_data2["is_in_person"] - influence_man_data2[
    "is_in_person"].min()) / (influence_man_data2["is_in_person"].max() - influence_man_data2["is_in_person"].min())
influence_man_data2["not_in_person"] = (influence_man_data2["not_in_person"] - influence_man_data2[
    "not_in_person"].min()) / (influence_man_data2["not_in_person"].max() - influence_man_data2["not_in_person"].min())
influence_man_data2["influence_val"] = (influence_man_data2["influence_val"] - influence_man_data2[
    "influence_val"].min()) / (influence_man_data2["influence_val"].max() - influence_man_data2["influence_val"].min())


# %%
def temp2(datas):
    K = np.power(np.sum(pow(datas, 2)), 0.5)
    for i in range(0, K.size):
        for j in range(0, datas[i].size):
            datas[i, j] = datas[i, j] / K[i]  # 套用矩阵标准化的公式
    return datas


# %%
influence_man_data2["influence_val"] = influence_man_data2["influence_val"] / np.sum(
    pow(influence_man_data2["influence_val"], 2))
influence_man_data3 = influence_man_data2
influence_man_data3.to_csv('./influence_man_data3.csv', index=False)

# %%

