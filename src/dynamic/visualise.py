import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dynamic_graph import DynamicGraphBuilder
# 1. Load Data
DF_PATH = "/Users/007vd/Downloads/DAU/Split_Graph_dl/data/simple_mlp/return_dataframe/return_df"
df = pd.read_csv(DF_PATH)
df = df.drop(columns=[col for col in ["Date", "Unnamed: 0"] if col in df.columns])
# Get asset names
assets = list(df.columns)
N = len(assets)
# 2. Get a sample window of returns
window_size = 30
momentum_window = 20
returns = df.values
sample_window = returns[-window_size:]  # Shape: (30, N)
sample_tensor = torch.tensor(sample_window, dtype=torch.float32)
# 3. Build Dynamic Graph
builder = DynamicGraphBuilder(momentum_window=momentum_window, top_k=3)
adj_matrix = builder.build_graph(sample_tensor)
# Extract momentum directly to visualize
recent_returns = sample_tensor[-momentum_window:]
momentum = torch.prod(1 + recent_returns, dim=0) - 1.0
momentum_np = momentum.numpy()
adj_np = adj_matrix.numpy()
# 4. Metrics
num_edges = np.count_nonzero(adj_np)
density = num_edges / (N * N)
degrees = np.sum(adj_np > 0, axis=1)
print(f"Number of nodes (assets): {N}")
print(f"Number of edges: {num_edges}")
print(f"Graph density: {density:.4f}")
# 5. Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Plot 1: Momentum Vector
axes[0].bar(assets, momentum_np, color='skyblue')
axes[0].set_title(f"Momentum Vector (Last {momentum_window} days)")
axes[0].set_ylabel("Cumulative Return")
axes[0].tick_params(axis='x', rotation=45)
# Plot 2: Dynamic Adjacency Heatmap
sns.heatmap(adj_np, ax=axes[1], cmap="YlGnBu", xticklabels=assets, yticklabels=assets)
axes[1].set_title(f"Dynamic Adjacency Matrix (Top-3)")
# Plot 3: Degree Distribution
axes[2].hist(degrees, bins=np.arange(0, N+2)-0.5, edgecolor='black', alpha=0.7)
axes[2].set_title("Degree Distribution (Out-degree)")
axes[2].set_xlabel("Degree")
axes[2].set_ylabel("Count")
axes[2].set_xticks(range(N+1))
plt.tight_layout()
plot_path = "/Users/007vd/Downloads/DAU/Split_Graph_dl/src/dynamic/dynamic_graph_visualization.png"
plt.savefig(plot_path)
print(f"\nVisualization saved to {plot_path}")
plt.show()
