# %%
import numpy as np
import pandas as pd

# %%
influence_data = pd.read_csv('./influence_data.csv', encoding='gbk')
data_by_year = pd.read_csv('./data_by_year.csv')
influence_man = influence_data["influencer_name"].value_counts()  # 统计影响者出现的次数
# %%
# 删除影响力小于20的影响者
influence_man = influence_man[influence_man > 100]
influence_man = pd.DataFrame({'person': influence_man.index, 'influence_value': influence_man.values})
# %%
# 获取主要影响者全部的数据并且保存到csv
influence_man_data = influence_data[influence_data['influencer_name'].isin(influence_man["person"])]
influence_man_data = influence_man_data.sample(1000)  # 随机抽样 保证样本比例不变，降低数据量
influence_man_data.index = np.arange(influence_man_data.shape[0])
# influence_man_data.to_csv('./influence_man_full_data.csv', index=0)
print(influence_man_data)  # 主要影响者的全部数据
# %%
influence_name_data = pd.DataFrame(influence_man_data["influencer_name"].drop_duplicates())
influence_name_data.columns = ["person"]
follower_name_data = pd.DataFrame(influence_man_data["follower_name"].drop_duplicates())
follower_name_data.columns = ["person"]
influence_man_nodes = pd.concat([influence_name_data, follower_name_data], axis=0)
influence_man_nodes = influence_man_nodes.drop_duplicates()
influence_man_nodes.index = np.arange(influence_man_nodes.shape[0])
print(influence_man_nodes)  # 所有歌手
# %%
from pyecharts import options as opts
from pyecharts.charts import Graph
# %%
nodes_data = []
# 节点数据
for i in np.arange(len(influence_man_nodes)):
    nodes_data.append(opts.GraphNode(name=influence_man_nodes["person"][i], symbol_size=10))

links_data = []
for i in np.arange(len(influence_man_data)):
    links_data.append(
        opts.GraphLink(source=influence_man_data["follower_name"][i],
                       target=influence_man_data["influencer_name"][i], value=2))


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
        # title_opts=opts.TitleOpts(title="Graph-GraphNode-GraphLink-WithEdgeLabel"),

    ).render("graph_with_edge_options.html")
)