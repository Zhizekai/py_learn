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
def get_influence_man_nodes():
    influence_name_data = pd.DataFrame(influence_man_data["influencer_name"].drop_duplicates())
    influence_name_data.columns = ["person"]
    follower_name_data = pd.DataFrame(influence_man_data["follower_name"].drop_duplicates())
    follower_name_data.columns = ["person"]
    nodes = pd.concat([influence_name_data, follower_name_data], axis=0)
    nodes = nodes.drop_duplicates()
    nodes.index = np.arange(nodes.shape[0])
    return nodes


influence_man_nodes = get_influence_man_nodes()

# %%
from pyecharts import options as opts
from pyecharts.charts import Graph


# %%
def generate_nodes_links():
    nodes_data1 = []
    # 节点数据
    for i in np.arange(len(influence_man_nodes)):
        nodes_data1.append(opts.GraphNode(name=influence_man_nodes["person"][i], symbol_size=10))
    links_data1 = []
    for i in np.arange(len(influence_man_data)):
        links_data1.append(opts.GraphLink(source=influence_man_data["follower_name"][i],
                                          target=influence_man_data["influencer_name"][i], value=2))
    return nodes_data, links_data


nodes_data, links_data = generate_nodes_links()

# %%

c = (
    Graph(
        init_opts=opts.InitOpts(width="100%", height="700px",
                                renderer="canvas"# 渲染模式 svg 或 canvas，即 RenderType.CANVAS 或 RenderType.SVG
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
