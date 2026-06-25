import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

df_path = "/Users/007vd/Downloads/DAU/Split_Graph_dl/data/simple_mlp/return_dataframe/return_df"
print("Loading data from:", df_path)
df = pd.read_csv(df_path)
df = df.drop(columns=[col for col in ["Date", "Unnamed: 0"] if col in df.columns])
stocks = list(df.columns)
num_assets = len(stocks)
print("Dataframe shape:", df.shape)
print(f"Number of stocks: {num_assets}")
print("Stocks:", stocks)


df_standardized = (df - df.mean()) / df.std()
returns_std = df_standardized.values
returns_raw = df.values
T, N = returns_std.shape

window_size = 30
X = []
original_returns=[]
for i in range(window_size, T + 1):
    window = returns_std[i - window_size:i]
    X.append(window)
    original_returns.append(returns_raw[i])
X = np.array(X)  
original_returns=np.array(original_returns)
X_tensor = torch.tensor(X, dtype=torch.float32)
original_returns=torch.tensor(original_returns,dtype=torch.float32)
# original_returns = torch.tensor(returns_raw[window_size - 1:], dtype=torch.float32)
print(f"\nWindowed input tensor shape: {X_tensor.shape}")
print(f"Target returns tensor shape: {original_returns.shape}")


class HistoricalState(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super().__init__()
        self.W1 = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.Ve = nn.Parameter(torch.randn(hidden_dim))
    def forward(self, H, X_last):
        B_total, L, Hdim = H.shape
        H_last = H[:, -1, :].unsqueeze(1).repeat(1, L, 1)
        concat = torch.cat([H, H_last], dim=-1)
        term1 = self.W1(concat)
        term2 = self.W2(X_last).unsqueeze(1)
        e = torch.tanh(term1 + term2)
        scores = torch.matmul(e, self.Ve)
        alpha = torch.softmax(scores, dim=1)
        E = (alpha.unsqueeze(-1) * H).sum(dim=1)
        return E, alpha
    

class StockEncoder(nn.Module):
    def __init__(self, num_assets, hidden_dim=64):
        super().__init__()
        self.num_assets = num_assets
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.attn = HistoricalState(hidden_dim, 1)
    def forward(self, x):
        B = x.shape[0]
        x = x.permute(0, 2, 1)
        x = x.reshape(B * self.num_assets, -1, 1)
        H, _ = self.lstm(x) 
        X_last = x[:, -1, :]
        E, alpha = self.attn(H, X_last) 
        E = E.view(B, self.num_assets, self.hidden_dim)
        return E, alpha
    

class StockPortfolioLSTM(nn.Module):
    def __init__(self, num_assets=15, hidden_dim=64):
        super().__init__()
        self.encoder = StockEncoder(num_assets=num_assets, hidden_dim=hidden_dim)
        # self.mlp = nn.Sequential(
        #     nn.Linear(hidden_dim, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 1)
        # )
        self.head=nn.Linear(hidden_dim,1)
    def forward(self, x):
        E, alpha = self.encoder(x)
        # logits = self.mlp(E).squeeze(-1)
        # weights = F.softmax(logits, dim=1)
        # return weights, E

        pred_returns = self.head(E)
        pred_returns = pred_returns.squeeze(-1)
        return pred_returns,E

        
    

hidden_dim = 64
model = StockPortfolioLSTM(num_assets=num_assets, hidden_dim=hidden_dim)

def compute_similarity_stats(embeddings_matrix):
   
    normalized_embeddings = F.normalize(embeddings_matrix, p=2, dim=1)
    cosine_sim_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t()).numpy()
    
    
    diffs = embeddings_matrix.unsqueeze(1) - embeddings_matrix.unsqueeze(0)
    euclidean_dist_matrix = torch.norm(diffs, p=2, dim=2).numpy()
    
    mask = np.eye(num_assets, dtype=bool)
    distinct_cosine_sims = cosine_sim_matrix[~mask]
    distinct_euclidean_dists = euclidean_dist_matrix[~mask]
    
    return {
        'cos_matrix': cosine_sim_matrix,
        'dist_matrix': euclidean_dist_matrix,
        'cos_mean': distinct_cosine_sims.mean(),
        'cos_min': distinct_cosine_sims.min(),
        'cos_max': distinct_cosine_sims.max(),
        'cos_std': distinct_cosine_sims.std(),
        'dist_mean': distinct_euclidean_dists.mean(),
        'dist_min': distinct_euclidean_dists.min(),
        'dist_max': distinct_euclidean_dists.max(),
        'dist_std': distinct_euclidean_dists.std(),
    }


model.eval()
with torch.no_grad():
    _, untrained_embeddings_seq = model(X_tensor)
untrained_last_day = untrained_embeddings_seq[-1]
untrained_stats = compute_similarity_stats(untrained_last_day)
print("\n--- Untrained Embedding Comparison (Last Day) ---")
print(f"Cosine Similarity Mean: {untrained_stats['cos_mean']:.6f} | Max: {untrained_stats['cos_max']:.6f} | Std: {untrained_stats['cos_std']:.6f}")
print(f"Euclidean Distance Mean: {untrained_stats['dist_mean']:.6f} | Min: {untrained_stats['dist_min']:.6f}")


def sharpe_loss(weights, future_returns):
    portfolio_returns = (weights * future_returns).sum(dim=1)
    mean_return = portfolio_returns.mean()
    std_return = portfolio_returns.std()
    return - (mean_return / (std_return + 1e-8))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("\n--- Training StockPortfolioLSTM model ---")
model.train()
num_epochs = 15
for epoch in range(num_epochs):
    optimizer.zero_grad()
    pred_returns, E = model(X_tensor)
    # loss = sharpe_loss(weights, original_returns)
    loss=criterion(pred_returns,original_returns)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | Sharpe Loss: {loss.item():.4f}")


model.eval()
with torch.no_grad():
    _, trained_embeddings_seq = model(X_tensor)
trained_last_day = trained_embeddings_seq[-1]
trained_stats = compute_similarity_stats(trained_last_day)
print("\n--- Trained Embedding Comparison (Last Day) ---")
print(f"Cosine Similarity Mean: {trained_stats['cos_mean']:.6f} | Max: {trained_stats['cos_max']:.6f} | Std: {trained_stats['cos_std']:.6f}")
print(f"Euclidean Distance Mean: {trained_stats['dist_mean']:.6f} | Min: {trained_stats['dist_min']:.6f}")

# assert trained_stats['cos_max'] < 0.999, f"Trained embeddings are too similar! Max similarity: {trained_stats['cos_max']:.4f}"
# assert trained_stats['cos_std'] > 0.05, f"Trained embeddings don't have enough variance! Std similarity: {trained_stats['cos_std']:.4f}"
# print("\nSuccess: Embeddings have been verified to be highly distinct and differentiated after training!")

print("\nTrained Pairwise Cosine Similarity Matrix:")
sim_df = pd.DataFrame(trained_stats['cos_matrix'], index=stocks, columns=stocks)
print(sim_df.round(4).to_string())

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

im1 = axes[0, 0].imshow(untrained_stats['cos_matrix'], cmap='coolwarm', vmin=-1.0, vmax=1.0)
axes[0, 0].set_xticks(np.arange(num_assets))
axes[0, 0].set_yticks(np.arange(num_assets))
axes[0, 0].set_xticklabels(stocks, rotation=45, ha='right')
axes[0, 0].set_yticklabels(stocks)
axes[0, 0].set_title(f"Untrained Cosine Similarity\n(Mean: {untrained_stats['cos_mean']:.3f}, Std: {untrained_stats['cos_std']:.3f})")
fig.colorbar(im1, ax=axes[0, 0], shrink=0.7)

X_ut_centered = untrained_last_day - untrained_last_day.mean(dim=0, keepdim=True)
_, _, V_ut = torch.pca_lowrank(X_ut_centered, q=2)
coords_ut = torch.matmul(X_ut_centered, V_ut[:, :2]).numpy()
axes[0, 1].scatter(coords_ut[:, 0], coords_ut[:, 1], color='#f43f5e', s=120, edgecolors='black', alpha=0.8)
for i, stock in enumerate(stocks):
    axes[0, 1].annotate(stock, (coords_ut[i, 0], coords_ut[i, 1]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, weight='bold')
axes[0, 1].set_title("Untrained Embeddings PCA Projection")
axes[0, 1].grid(True, linestyle='--', alpha=0.5)


im2 = axes[1, 0].imshow(trained_stats['cos_matrix'], cmap='coolwarm', vmin=-1.0, vmax=1.0)
axes[1, 0].set_xticks(np.arange(num_assets))
axes[1, 0].set_yticks(np.arange(num_assets))
axes[1, 0].set_xticklabels(stocks, rotation=45, ha='right')
axes[1, 0].set_yticklabels(stocks)
axes[1, 0].set_title(f"Trained Cosine Similarity\n(Mean: {trained_stats['cos_mean']:.3f}, Std: {trained_stats['cos_std']:.3f})")
fig.colorbar(im2, ax=axes[1, 0], shrink=0.7)

X_t_centered = trained_last_day - trained_last_day.mean(dim=0, keepdim=True)
_, _, V_t = torch.pca_lowrank(X_t_centered, q=2)
coords_t = torch.matmul(X_t_centered, V_t[:, :2]).numpy()
axes[1, 1].scatter(coords_t[:, 0], coords_t[:, 1], color='#10b981', s=120, edgecolors='black', alpha=0.8)
for i, stock in enumerate(stocks):
    axes[1, 1].annotate(stock, (coords_t[i, 0], coords_t[i, 1]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, weight='bold')
axes[1, 1].set_title("Trained Embeddings PCA Projection")
axes[1, 1].grid(True, linestyle='--', alpha=0.5)
plt.suptitle("Stock Embeddings Comparison: Untrained vs. Trained LSTM+Attention", fontsize=16, weight='bold')
plt.tight_layout()
output_plot = Path("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/encoddings.png")
plt.savefig(output_plot, dpi=150)
print(f"\nSaved comparison plot to: {output_plot}")
plt.close()
