#%%
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import itertools
from torch.utils.data import Dataset, DataLoader
from mgan import MGANPortfolio

#%%
df = pd.read_csv("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/simple_mlp/return_dataframe/return_df")
df = df.drop(columns=[col for col in ["Date", "Unnamed: 0"] if col in df.columns])
stocks = list(df.columns)

returns_raw = df.values
split_index = int(0.8 * len(returns_raw))
train_raw   = returns_raw[:split_index]
test_raw    = returns_raw[split_index:]

train_mean  = train_raw.mean(axis=0)
train_std   = train_raw.std(axis=0)
train_std_r = (train_raw - train_mean) / train_std
test_std_r  = (test_raw  - train_mean) / train_std

window_size = 30

def make_windows(std_returns, raw_returns, window):
    X, y = [], []
    for i in range(window, len(std_returns)):
        X.append(std_returns[i - window:i])
        y.append(raw_returns[i])
    return np.array(X), np.array(y)

x_train, y_train = make_windows(train_std_r, train_raw, window_size)
x_test,  y_test  = make_windows(test_std_r,  test_raw,  window_size)

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test  = torch.tensor(x_test,  dtype=torch.float32)
y_test  = torch.tensor(y_test,  dtype=torch.float32)

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels   = labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self, index):
        return self.features[index], self.labels[index]

train_dataset = CustomDataset(x_train, y_train)
test_dataset  = CustomDataset(x_test,  y_test)
train_loader  = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
test_loader   = DataLoader(test_dataset,  batch_size=len(test_dataset),  shuffle=False)

sector_adj   = torch.load("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/static/A_sector.pt")
industry_adj = torch.load("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/static/A_industry.pt")
theme_adj    = torch.load("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/static/A_theme.pt")

#%%
def sharpe_loss(weights, future_returns):
    port_r = (weights * future_returns).sum(dim=1)
    return -(port_r.mean() / (port_r.std() + 1e-8))

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

#%%
def run_trial(hidden_dim, num_heads, lr, weight_decay, num_epochs, eta_min):
    """Train one configuration, return best val Sharpe and best model state."""

    model = MGANPortfolio(
        sector_adj=sector_adj,
        industry_adj=industry_adj,
        theme_adj=theme_adj,
        num_assets=15,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=eta_min
    )

    best_val   = -np.inf
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        X_full, y_full = next(iter(train_loader))
        weights = model(X_full)
        loss    = sharpe_loss(weights, y_full)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            X_val, y_val = next(iter(test_loader))
            val_w        = model(X_val)
            val_port_r   = (val_w * y_val).sum(dim=1)
        val_sharpe = annualized_sharpe(val_port_r).item()

        if val_sharpe > best_val:
            best_val   = val_sharpe
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    return best_val, best_state, model

#%%
# ── Hyperparameter grid ───────────────────────────────────────────────────────
# Keep it tractable: 3×2×3×2×2 = 72 trials
# Adjust ranges based on your compute budget
param_grid = {
    "hidden_dim"   : [16, 32, 64],
    "num_heads"    : [2, 4],
    "lr"           : [5e-4, 3e-4, 1e-4],
    "weight_decay" : [1e-5, 1e-4],
    "num_epochs"   : [150],        # fixed — enough for convergence
    "eta_min"      : [1e-5, 5e-5],
}

keys   = list(param_grid.keys())
combos = list(itertools.product(*param_grid.values()))

print(f"Total trials: {len(combos)}")
print("=" * 70)

results     = []
global_best_sharpe = -np.inf
global_best_state  = None
global_best_params = None
global_best_model_cfg = None  # (hidden_dim, num_heads) to rebuild model

for i, values in enumerate(combos):
    params = dict(zip(keys, values))

    print(f"\nTrial {i+1}/{len(combos)} | "
          f"hidden={params['hidden_dim']} heads={params['num_heads']} "
          f"lr={params['lr']:.0e} wd={params['weight_decay']:.0e} "
          f"eta_min={params['eta_min']:.0e}")

    best_val, best_state, model = run_trial(**params)

    results.append({**params, "best_val_sharpe": best_val})
    print(f"  → Best Val Sharpe: {best_val:.4f}")

    if best_val > global_best_sharpe:
        global_best_sharpe    = best_val
        global_best_state     = best_state
        global_best_params    = params
        global_best_model_cfg = (params["hidden_dim"], params["num_heads"])
        print(f"  ★ New global best: {global_best_sharpe:.4f}")

#%%
# ── Results summary ───────────────────────────────────────────────────────────
results_df = pd.DataFrame(results).sort_values("best_val_sharpe", ascending=False)
print("\n=== TOP 10 CONFIGURATIONS ===")
print(results_df.head(10).to_string(index=False))

#%%
# ── Final evaluation with best model ─────────────────────────────────────────
print(f"\n=== BEST CONFIG ===")
for k, v in global_best_params.items():
    print(f"  {k}: {v}")
print(f"  best_val_sharpe: {global_best_sharpe:.4f}")

# Rebuild best model and load weights
best_model = MGANPortfolio(
    sector_adj=sector_adj,
    industry_adj=industry_adj,
    theme_adj=theme_adj,
    num_assets=15,
    hidden_dim=global_best_model_cfg[0],
    num_heads=global_best_model_cfg[1],
)
best_model.load_state_dict(global_best_state)
best_model.eval()

with torch.no_grad():
    X_val, y_val = next(iter(test_loader))
    val_w        = best_model(X_val)
    val_port_r   = (val_w * y_val).sum(dim=1)

print("\n=== FINAL VALIDATION METRICS (best model) ===")
print(f"Sharpe     : {annualized_sharpe(val_port_r):.4f}")
print(f"Volatility : {annualized_volatility(val_port_r):.4f}")
print(f"Sortino    : {sortino_ratio(val_port_r):.4f}")
print(f"Max DD     : {max_drawdown(val_port_r):.4f}")
print(f"Cum Return : {cumulative_return(val_port_r):.4f}")

# Equal weight baseline
equal_w   = torch.ones(15) / 15
eq_port_r = (y_val * equal_w).sum(dim=1)
print(f"\n=== EQUAL WEIGHT BASELINE ===")
print(f"Sharpe     : {annualized_sharpe(eq_port_r):.4f}")
print(f"Cum Return : {cumulative_return(eq_port_r):.4f}")

#%%
# ── Save best model weights ───────────────────────────────────────────────────
torch.save(global_best_state, "/Users/007vd/Downloads/DAU/Split_Graph_dl/data/best_mgan.pt")
print("\nBest model saved to best_mgan.pt")

#%%
# ── Plot: val Sharpe distribution across trials ───────────────────────────────
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sharpes = [r["best_val_sharpe"] for r in results]
plt.hist(sharpes, bins=20, edgecolor="black")
plt.axvline(global_best_sharpe, color="red", linestyle="--", label=f"Best: {global_best_sharpe:.3f}")
plt.xlabel("Val Sharpe")
plt.ylabel("Count")
plt.title("Val Sharpe Distribution Across Trials")
plt.legend()

plt.subplot(1, 2, 2)
for hd in param_grid["hidden_dim"]:
    subset = [r["best_val_sharpe"] for r in results if r["hidden_dim"] == hd]
    plt.plot(sorted(subset, reverse=True), label=f"hidden={hd}")
plt.xlabel("Trial rank")
plt.ylabel("Val Sharpe")
plt.title("Val Sharpe by Hidden Dim")
plt.legend()
plt.tight_layout()
plt.show()