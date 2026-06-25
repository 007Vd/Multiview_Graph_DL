#%% 
import numpy as np 
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset,DataLoader


torch.manual_seed(42)
np.random.seed(42)

df=pd.read_csv("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/simple_mlp/return_dataframe/return_df")
df = df.drop(columns=[col for col in ["Date", "Unnamed: 0"] if col in df.columns])
df
stocks = list(df.columns)
# %%
returns_raw=df.values
split_index=int(0.8*(len(returns_raw)))
train_returns=returns_raw[:split_index]
test_returns=returns_raw[split_index:]
train_mean = train_returns.mean(axis=0)
train_std  = train_returns.std(axis=0)

train_returns_std = (
    train_returns - train_mean
) / train_std

test_returns_std = (
    test_returns - train_mean
) / train_std

window_size=30
x_train=[]
y_train=[]
for i in range(window_size,len(train_returns_std)):
    x_train.append(train_returns_std[i-window_size:i])
    y_train.append(train_returns_std[i])
x_train = np.array(x_train)
y_train = np.array(y_train)

x_test=[]
y_test=[]
for i in range(window_size,len(test_returns_std)):
    x_test.append(test_returns_std[i-window_size:i])
    y_test.append(test_returns_std[i])

x_test = np.array(x_test)
y_test = np.array(y_test)

x_train=torch.tensor(x_train, dtype=torch.float32)
y_train=torch.tensor(y_train, dtype=torch.float32)
x_test=torch.tensor(x_test, dtype=torch.float32)
y_test=torch.tensor(y_test, dtype=torch.float32)
# %%
class CustomDataset(Dataset):
    def __init__(self,features,labels):
        self.features=features
        self.labels=labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        # return super().__getitem__(index)
        return self.features[index],self.labels[index]
    
train_dataset=CustomDataset(x_train,y_train)
test_dataset=CustomDataset(x_test,y_test)

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)

# %%
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

hidden_dim=64
num_assets=15
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
    _, untrained_embeddings_seq = model(x_train)
untrained_last_day = untrained_embeddings_seq[-1]
untrained_stats = compute_similarity_stats(untrained_last_day)
print("\n--- Untrained Embedding Comparison (Last Day) ---")
print(f"Cosine Similarity Mean: {untrained_stats['cos_mean']:.6f} | Max: {untrained_stats['cos_max']:.6f} | Std: {untrained_stats['cos_std']:.6f}")
print(f"Euclidean Distance Mean: {untrained_stats['dist_mean']:.6f} | Min: {untrained_stats['dist_min']:.6f}")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    # weight_decay=1e-4
)
num_epochs=15
print("\n--- Training StockPortfolioLSTM model ---")
train_losses = []
val_losses = []
for epoch in range(num_epochs):

    model.train()

    train_loss = 0

    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()

        pred_returns,_ = model(X_batch)

        loss = criterion(
            pred_returns,
            y_batch
        )

        loss.backward()
#         torch.nn.utils.clip_grad_norm_(
#     model.parameters(),
#     max_norm=1.0
# )

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            pred_returns,_ = model(X_batch)

            loss = criterion(
            pred_returns,
            y_batch
        )

            val_loss += loss.item()

    val_loss /= len(test_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(
    f"Epoch {epoch+1} | "
    f"Train Loss: {train_loss:.6f} | "
    f"Val Loss: {val_loss:.6f}"
    )


plt.figure(figsize=(10,5))

plt.plot(
    train_losses,
    label="Train Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)

plt.show()

model.eval()
with torch.no_grad():
    _, trained_embeddings_seq = model(x_train)
trained_last_day = trained_embeddings_seq[-1]
trained_stats = compute_similarity_stats(trained_last_day)
print("\n--- Trained Embedding Comparison (Last Day) ---")
print(f"Cosine Similarity Mean: {trained_stats['cos_mean']:.6f} | Max: {trained_stats['cos_max']:.6f} | Std: {trained_stats['cos_std']:.6f}")
print(f"Euclidean Distance Mean: {trained_stats['dist_mean']:.6f} | Min: {trained_stats['dist_min']:.6f}")
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

# %%
