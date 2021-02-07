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
influence_man_data = influence_man_data.sample(2000)  # 随机抽样 保证样本比例不变，降低数据量
influence_man_data.index = np.arange(influence_man_data.shape[0])
# influence_man_data.to_csv('./influence_man_full_data.csv', index=0)
print(influence_man_data)  # 主要影响者的全部数据


# %%
def get_influence_man_nodes():
    # influence_name_data = pd.DataFrame(influence_man_data["influencer_name"].drop_duplicates())
    influence_name_data = influence_man_data.loc[:, ["influencer_name", "influencer_main_genre"]].drop_duplicates(
        "influencer_name")
    influence_name_data.columns = ["person", "main_genre"]
    # follower_name_data = pd.DataFrame(influence_man_data["follower_name"].drop_duplicates())
    follower_name_data = influence_man_data.loc[:, ["follower_name", "follower_main_genre"]].drop_duplicates(
        "follower_name")
    follower_name_data.columns = ["person", "main_genre"]
    nodes = pd.concat([influence_name_data, follower_name_data], axis=0)
    nodes = nodes.drop_duplicates()
    nodes["main_genre"] = nodes["main_genre"].map({
        "Pop/Rock": 0,
        "R&B;": 1,
        "Country": 2,
        "Jazz": 3,
        "Electronic": 4,
        "Vocal": 5,
        "Blues": 6,
        "Folk": 7,
        "Reggae": 8,
        "Latin": 9,
        "Religious": 10,
        "New Age": 11,
        "International": 12,
        "Classical": 13,
        "Avant-Garde": 14,

    })
    nodes.index = np.arange(nodes.shape[0])
    return nodes

influence_man_nodes = get_influence_man_nodes()

# %%
influence_man2 = influence_data["influencer_name"].value_counts()  # 统计影响者出现的次数
influence_man2 = pd.DataFrame({'person': influence_man2.index, 'influence_value': influence_man2.values})
influence_man_nodes3 = pd.merge(influence_man2,influence_man_nodes,on=["person"],how='right')
influence_man_nodes = influence_man_nodes3.fillna(1)


# %%
from pyecharts import options as opts
from pyecharts.charts import Graph


# %%
def generate_nodes_links():
    nodes_data1 = []  # 节点数据
    for i in np.arange(len(influence_man_nodes)):
        nodes_data1.append(opts.GraphNode(name=influence_man_nodes["person"][i],
                                          symbol_size=30 if int(influence_man_nodes["influence_value"][i]) > 50 else 10,
                                          category=int(influence_man_nodes["main_genre"][i])
                                          ))
        # nodes_data1.append({"name": influence_man_nodes["person"][i],
        #                     # "symbol_size": 10,
        #                     "category": int(influence_man_nodes["main_genre"][i])
        #                     })
    links_data1 = []  # 关系数据 边数据
    for i in np.arange(len(influence_man_data)):
        links_data1.append(opts.GraphLink(source=influence_man_data["follower_name"][i],
                                          target=influence_man_data["influencer_name"][i],
                                          value=2))
    return nodes_data1, links_data1


nodes_data, links_data = generate_nodes_links()

# %%
cat = [{"name": "Pop/Rock", "symbol": "circle"}]
cc = {
    "R&B;": 1,
    "Country": 2,
    "Jazz": 3,
    "Electronic": 4,
    "Vocal": 5,
    "Blues": 6,
    "Folk": 7,
    "Reggae": 8,
    "Latin": 9,
    "Religious": 10,
    "New Age": 11,
    "International": 12,
    "Classical": 13,
    "Avant-Garde": 14,

}
for key in cc:
    cat.append({"name": key, "symbol": "circle"})
# %%

c = (
    Graph(
        init_opts=opts.InitOpts(width="100%", height="700px",
                                renderer="canvas"  # 渲染模式 svg 或 canvas，即 RenderType.CANVAS 或 RenderType.SVG
                                ),

    ).add(
        "",
        nodes_data,
        links_data,
        categories=cat,
        repulsion=50,
        linestyle_opts=opts.LineStyleOpts(curve=0.2),
        label_opts=opts.LabelOpts(is_show=False),
        edge_symbol=['none', 'arrow'],
        is_draggable=True
    ).set_global_opts(
        legend_opts=opts.LegendOpts(is_show=False),
        # title_opts=opts.TitleOpts(title="Graph-GraphNode-GraphLink-WithEdgeLabel"),
    ).render("graph_with_edge_options.html")
)
# %%
#
nodes = [
    {"name": "木", "category": 0,"symbolSize": 10},
    {"name": "火", "category": 1,"symbolSize": 30},
    {"name": "土", "category": 2,"symbolSize": 50},
    {"name": "金", "category": 3},
    {"name": "水", "category": 4},
]

# node的category通过index查找对应的类别
# 不同类别的symbol会自动渲染成不同颜色
# 还可以让不同类别节点设置不同的symbol符号
# 如果类别symbol和节点symbol冲突，节点symbol优先
categories = [
    {"name": "木类节点", "symbol": "rect"},
    {"name": "火类节点", "symbol": "roundRect"},
    {"name": "土类节点", "symbol": "triangle"},
    {"name": "金类节点", "symbol": "diamond"},
    {"name": "水类节点", "symbol": "pin"},

]

# 节点连线
links = []
for i in nodes:
    for j in nodes:
        links.append({"source": i.get("name"), "target": j.get("name")})

c = (
    Graph()
        .add("a", nodes, links, categories=cat,
             repulsion=50,
             linestyle_opts=opts.LineStyleOpts(curve=0.2),
             label_opts=opts.LabelOpts(is_show=False),
             edge_symbol=['none', 'arrow'])
        .set_global_opts(
        legend_opts=opts.LegendOpts(is_show=False),
        # title_opts=opts.TitleOpts(title="Graph-GraphNode-GraphLink-WithEdgeLabel"),
    )
).render("graph_with_edge_options1.html")
