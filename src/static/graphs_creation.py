#%%
import torch
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
with open("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/stock_metadata.json","r") as f:
    metadata=json.load(f)

N=len(metadata)
print(f"length of metadata is : {N}")
# %%
A_sector=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if metadata[i]["sector"]==metadata[j]["sector"]:
            A_sector[i,j]=1


A_industry=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if metadata[i]["industry"]==metadata[j]["industry"]:
            A_sector[i,j]=1

def jaccard_similarity(a,b):
    a=set(a)
    b=set(b)
    intersection=len(a&b)
    union=len(a|b)

    return intersection/union
A_theme=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        A_theme[i,j]=jaccard_similarity(metadata[i]["themes"], metadata[j]["themes"])

np.fill_diagonal(A_sector,1)
np.fill_diagonal(A_industry,1)
np.fill_diagonal(A_theme,1)
# %%
A_sector=torch.tensor(A_sector,dtype=torch.float32)
A_industry=torch.tensor(A_industry,dtype=torch.float32)
A_theme=torch.tensor(A_theme,dtype=torch.float32)

# %%
print("Sector edges:",(A_sector > 0).sum())

print("Industry edges:",(A_industry > 0).sum())

print("Theme edges:",(A_theme > 0).sum())
# %%
STATIC_GRAPH_DIR=Path("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/static")
STATIC_GRAPH_DIR.mkdir(parents=True,exist_ok=True)

torch.save(A_sector,STATIC_GRAPH_DIR/"A_sector.pt")
torch.save(A_industry,STATIC_GRAPH_DIR/"A_industry.pt")
torch.save(A_theme,STATIC_GRAPH_DIR/"A_theme.pt")


#%%
tickers = [
    "AAPL","MSFT","NVDA","JPM","V",
    "JNJ","UNH","CVX","PG","WMT",
    "HD","CAT","HON","DIS","KO"
]

fig, ax = plt.subplots(
    1,
    3,
    figsize=(20,6)
)

for graph, title, axis in zip(
    [A_sector, A_industry, A_theme],
    ["Sector", "Industry", "Theme"],
    ax
):

    im = axis.imshow(
        graph.numpy(),
        cmap="Blues"
    )

    axis.set_title(title)

    axis.set_xticks(range(len(tickers)))
    axis.set_yticks(range(len(tickers)))

    axis.set_xticklabels(
        tickers,
        rotation=90
    )

    axis.set_yticklabels(
        tickers
    )

    plt.colorbar(
        im,
        ax=axis
    )

plt.tight_layout()
plt.show()

G = nx.Graph()

for ticker in tickers:
    G.add_node(ticker)

for i in range(len(tickers)):
    for j in range(i + 1, len(tickers)):

        if A_sector[i, j] > 0:

            G.add_edge(
                tickers[i],
                tickers[j]
            )

plt.figure(figsize=(10,8))

pos = nx.spring_layout(
    G,
    seed=42
)

nx.draw_networkx(
    G,
    pos,
    node_size=2500,
    font_size=10
)

plt.title("Sector Graph")
plt.axis("off")
plt.show()


G = nx.Graph()

for ticker in tickers:
    G.add_node(ticker)

for i in range(len(tickers)):
    for j in range(i + 1, len(tickers)):

        if A_industry[i, j] > 0:

            G.add_edge(
                tickers[i],
                tickers[j]
            )

plt.figure(figsize=(10,8))

pos = nx.spring_layout(
    G,
    seed=42
)

nx.draw_networkx(
    G,
    pos,
    node_size=2500,
    font_size=10
)

plt.title("Industry Graph")
plt.axis("off")
plt.show()


G = nx.Graph()

for ticker in tickers:
    G.add_node(ticker)

for i in range(len(tickers)):
    for j in range(i + 1, len(tickers)):

        weight = float(A_theme[i,j])

        if weight > 0:

            G.add_edge(
                tickers[i],
                tickers[j],
                weight=weight
            )

plt.figure(figsize=(12,10))

pos = nx.spring_layout(
    G,
    seed=42,
    k=1
)

edge_widths = [
    5 * G[u][v]["weight"]
    for u,v in G.edges()
]

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=2500
)

nx.draw_networkx_labels(
    G,
    pos
)

nx.draw_networkx_edges(
    G,
    pos,
    width=edge_widths
)

plt.title("Theme Graph")
plt.axis("off")
plt.show()