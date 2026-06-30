#%%
import numpy as np 
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from mgan import MGANPortfolio

# ── Reproducibility ───────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

#%%
df = pd.read_csv("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/simple_mlp/return_dataframe/return_df")
df = df.drop(columns=[col for col in ["Date", "Unnamed: 0"] if col in df.columns])
stocks = list(df.columns)

# %%
# ── FIX 1: Proper 70 / 15 / 15 split ─────────────────────
returns_raw = df.values
n           = len(returns_raw)
train_end   = int(0.70 * n)
val_end     = int(0.85 * n)

train_returns = returns_raw[:train_end]
val_returns   = returns_raw[train_end:val_end]
test_returns  = returns_raw[val_end:]

# Normalize with train stats only
train_mean = train_returns.mean(axis=0)
train_std  = train_returns.std(axis=0) + 1e-8

train_std_r = (train_returns - train_mean) / train_std
val_std_r   = (val_returns   - train_mean) / train_std
test_std_r  = (test_returns  - train_mean) / train_std

# ── FIX 2: Windowed sequences for all three splits ────────
def make_windows(std_r, raw_r, window_size=30):
    X, y = [], []
    for i in range(window_size, len(std_r)):
        X.append(std_r[i - window_size:i])
        y.append(raw_r[i])
    return (
        torch.tensor(np.array(X), dtype=torch.float32),
        torch.tensor(np.array(y), dtype=torch.float32),
    )

window_size = 30
x_train, y_train = make_windows(train_std_r, train_returns, window_size)
x_val,   y_val   = make_windows(val_std_r,   val_returns,   window_size)
x_test,  y_test  = make_windows(test_std_r,  test_returns,  window_size)

print(f"Train: {x_train.shape} | Val: {x_val.shape} | Test: {x_test.shape}")

# %%
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels   = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

# ── FIX 3: Shuffle train, keep val/test ordered ───────────
BATCH_SIZE = 64

train_dataset = CustomDataset(x_train, y_train)
val_dataset   = CustomDataset(x_val,   y_val)
test_dataset  = CustomDataset(x_test,  y_test)

train_loader = DataLoader(train_dataset, batch_size=len(train_dataset),          shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=len(val_dataset),    shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=len(test_dataset),   shuffle=False)

# ── Graph adjacency matrices ──────────────────────────────
sector_adj   = torch.load("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/static/A_sector.pt")
industry_adj = torch.load("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/static/A_industry.pt")
theme_adj    = torch.load("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/static/A_theme.pt")

print("sector  — shape:", sector_adj.shape,   "| nonzero rows:", (sector_adj.sum(1)   > 0).sum().item())
print("industry— shape:", industry_adj.shape, "| nonzero rows:", (industry_adj.sum(1) > 0).sum().item())
print("theme   — shape:", theme_adj.shape,    "| nonzero rows:", (theme_adj.sum(1)    > 0).sum().item())
print("sector density:",   sector_adj.mean().item())
print("industry density:", industry_adj.mean().item())
print("theme density:",    theme_adj.mean().item())

#%%
model = MGANPortfolio(
    sector_adj=sector_adj,
    industry_adj=industry_adj,
    theme_adj=theme_adj,
    num_assets=15,
    hidden_dim=16,
    num_heads=4,
)
print("Parameters:", sum(p.numel() for p in model.parameters()))

# ── Metric helpers ────────────────────────────────────────
def annualized_sharpe(returns):
    return (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252)

def annualized_volatility(returns):
    return returns.std() * np.sqrt(252)

def sortino_ratio(returns):
    downside = returns[returns < 0]
    if len(downside) == 0:
        return torch.tensor(float("inf"))
    return (returns.mean() / (downside.std() + 1e-8)) * np.sqrt(252)

def max_drawdown(returns):
    wealth      = torch.cumprod(1 + returns, dim=0)
    running_max = torch.cummax(wealth, dim=0)[0]
    return ((wealth - running_max) / running_max).min()

def cumulative_return(returns):
    return torch.cumprod(1 + returns, dim=0)[-1] - 1

def sharpe_loss(weights, future_returns):
    port_r = (weights * future_returns).sum(dim=1)
    return -(port_r.mean() / (port_r.std() + 1e-8))

# ── Optimizer & scheduler ─────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=5e-5)

# ── FIX 4: Training loop with minibatches + checkpointing ─
num_epochs  = 50
best_val    = -np.inf
best_path   = "best_model.pt"
train_sharpes, val_sharpes = [], []

for epoch in range(num_epochs):

    # ── Train (minibatches) ───────────────────────────────
    model.train()
    batch_sharpes = []
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        weights = model(X_batch)
        loss    = sharpe_loss(weights, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            port_r = (weights * y_batch).sum(dim=1)
            batch_sharpes.append(annualized_sharpe(port_r).item())

    scheduler.step()
    train_sharpe = float(np.mean(batch_sharpes))

    # ── Validate ──────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        X_val_full, y_val_full = next(iter(val_loader))
        val_w      = model(X_val_full)
        val_port_r = (val_w * y_val_full).sum(dim=1)
        val_sharpe = annualized_sharpe(val_port_r).item()

    # ── FIX 5: Save best checkpoint ───────────────────────
    if val_sharpe > best_val:
        best_val = val_sharpe
        torch.save(model.state_dict(), best_path)

    train_sharpes.append(train_sharpe)
    val_sharpes.append(val_sharpe)

    print(
        f"Epoch {epoch+1:3d}/{num_epochs} | "
        f"Train: {train_sharpe:.4f} | "
        f"Val: {val_sharpe:.4f} | "
        f"Best Val: {best_val:.4f}"
    )
    print(
        f"  Vol: {annualized_volatility(val_port_r):.4f} | "
        f"Sortino: {sortino_ratio(val_port_r):.4f} | "
        f"MaxDD: {max_drawdown(val_port_r):.4f} | "
        f"CumRet: {cumulative_return(val_port_r):.4f}"
    )

# ── FIX 6: Load best model for test evaluation ────────────
print(f"\nBest Val Sharpe reached: {best_val:.4f}")
model.load_state_dict(torch.load(best_path))
model.eval()

with torch.no_grad():
    X_test_full, y_test_full = next(iter(test_loader))
    test_w      = model(X_test_full)
    test_port_r = (test_w * y_test_full).sum(dim=1)

test_sharpe = annualized_sharpe(test_port_r)

print("\n── Test Set Results (best checkpoint) ──────────────")
print(f"  Sharpe:    {test_sharpe:.4f}")
print(f"  Vol:       {annualized_volatility(test_port_r):.4f}")
print(f"  Sortino:   {sortino_ratio(test_port_r):.4f}")
print(f"  MaxDD:     {max_drawdown(test_port_r):.4f}")
print(f"  CumRet:    {cumulative_return(test_port_r):.4f}")

avg_weights = test_w.mean(dim=0)
print(f"\nAvg weights: {avg_weights}")
print(f"Weight sum:  {avg_weights.sum():.4f}")
print(f"Weight std:  {test_w.std(dim=0)}")

# ── Equal-weight baseline (on TEST set) ───────────────────
equal_weights = torch.ones(15) / 15
eq_returns    = (y_test_full * equal_weights).sum(dim=1)
eq_sharpe     = annualized_sharpe(eq_returns)

print("\n── Equal Weight Baseline ────────────────────────────")
print(f"  Sharpe:    {eq_sharpe:.4f}")
print(f"  Vol:       {annualized_volatility(eq_returns):.4f}")
print(f"  Sortino:   {sortino_ratio(eq_returns):.4f}")
print(f"  MaxDD:     {max_drawdown(eq_returns):.4f}")
print(f"  CumRet:    {cumulative_return(eq_returns):.4f}")

# ── Plots ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot([-s for s in train_sharpes], label="Train Loss")
axes[0].plot([-s for s in val_sharpes],   label="Val Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (= −Sharpe)")
axes[0].set_title("Training vs Validation Loss")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(train_sharpes, label="Train Sharpe")
axes[1].plot(val_sharpes,   label="Val Sharpe")
axes[1].axhline(best_val,        color="green", linestyle="--", alpha=0.7, label=f"Best Val {best_val:.3f}")
axes[1].axhline(eq_sharpe.item(), color="red",   linestyle="--", alpha=0.7, label=f"Equal Weight {eq_sharpe:.3f}")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Annualized Sharpe")
axes[1].set_title("Training vs Validation Sharpe")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()