#%%
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import minimize

# ── Reproducibility ───────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# =============================================================
# 1. DATA
# =============================================================
df = pd.read_csv("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/simple_mlp/return_dataframe/return_df")
df = df.drop(columns=[col for col in ["Date", "Unnamed: 0"] if col in df.columns])
stocks     = list(df.columns)
NUM_ASSETS = len(stocks)

returns_raw = df.values
n           = len(returns_raw)
train_end   = int(0.70 * n)
val_end     = int(0.85 * n)

train_returns = returns_raw[:train_end]
val_returns   = returns_raw[train_end:val_end]
test_returns  = returns_raw[val_end:]

train_mean = train_returns.mean(axis=0)
train_std  = train_returns.std(axis=0) + 1e-8

train_std_r = (train_returns - train_mean) / train_std
val_std_r   = (val_returns   - train_mean) / train_std
test_std_r  = (test_returns  - train_mean) / train_std

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

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels   = labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# BATCH_SIZE   = 64
train_loader = DataLoader(CustomDataset(x_train, y_train), batch_size=len(y_train), shuffle=True)
val_loader   = DataLoader(CustomDataset(x_val,   y_val),   batch_size=len(y_val),   shuffle=False)
test_loader  = DataLoader(CustomDataset(x_test,  y_test),  batch_size=len(y_test),  shuffle=False)

# =============================================================
# 2. METRICS
# =============================================================
def annualized_sharpe(returns):
    r = returns if isinstance(returns, torch.Tensor) else torch.tensor(returns, dtype=torch.float32)
    return (r.mean() / (r.std() + 1e-8)) * np.sqrt(252)

def annualized_volatility(returns):
    r = returns if isinstance(returns, torch.Tensor) else torch.tensor(returns, dtype=torch.float32)
    return r.std() * np.sqrt(252)

def sortino_ratio(returns):
    r = returns if isinstance(returns, torch.Tensor) else torch.tensor(returns, dtype=torch.float32)
    downside = r[r < 0]
    if len(downside) == 0:
        return torch.tensor(float("inf"))
    return (r.mean() / (downside.std() + 1e-8)) * np.sqrt(252)

def max_drawdown(returns):
    r = returns if isinstance(returns, torch.Tensor) else torch.tensor(returns, dtype=torch.float32)
    wealth      = torch.cumprod(1 + r, dim=0)
    running_max = torch.cummax(wealth, dim=0)[0]
    return ((wealth - running_max) / running_max).min()

def cumulative_return(returns):
    r = returns if isinstance(returns, torch.Tensor) else torch.tensor(returns, dtype=torch.float32)
    return torch.cumprod(1 + r, dim=0)[-1] - 1

def compute_all_metrics(returns, name):
    r = returns if isinstance(returns, torch.Tensor) else torch.tensor(returns, dtype=torch.float32)
    return {
        "Strategy":  name,
        "Sharpe":    round(annualized_sharpe(r).item(),    4),
        "Vol":       round(annualized_volatility(r).item(), 4),
        "Sortino":   round(sortino_ratio(r).item(),         4),
        "MaxDD":     round(max_drawdown(r).item(),          4),
        "CumRet":    round(cumulative_return(r).item(),     4),
    }

def sharpe_loss(weights, future_returns):
    port_r = (weights * future_returns).sum(dim=1)
    return -(port_r.mean() / (port_r.std() + 1e-8))

# =============================================================
# 3. CLASSICAL BASELINES  (fitted on train_returns, evaluated on test_returns)
# =============================================================

# helper: portfolio returns given static weights
def port_returns_np(w, ret):
    return ret @ w          # (T, N) @ (N,) -> (T,)

def static_weights_to_tensor(w_np, test_ret_tensor):
    """Broadcast static weights across all test periods."""
    w = torch.tensor(w_np, dtype=torch.float32)
    # test_ret_tensor: (T, N)  →  port returns: (T,)
    return (test_ret_tensor * w).sum(dim=1)

# ── 3a. Equal Weight ──────────────────────────────────────
w_ew = np.ones(NUM_ASSETS) / NUM_ASSETS

# ── 3b. Mean-Variance (Max Sharpe) ───────────────────────
# Estimated from TRAIN data; long-only, fully invested
mu_train  = train_returns.mean(axis=0)
cov_train = np.cov(train_returns.T)

def neg_sharpe_np(w):
    port_r = mu_train @ w
    port_v = np.sqrt(w @ cov_train @ w + 1e-10)
    return -port_r / port_v

constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
bounds      = [(0, 1)] * NUM_ASSETS
w0          = np.ones(NUM_ASSETS) / NUM_ASSETS

res_mv = minimize(neg_sharpe_np, w0, method="SLSQP",
                  bounds=bounds, constraints=constraints,
                  options={"maxiter": 1000, "ftol": 1e-12})
w_mv = res_mv.x

# ── 3c. Minimum Variance ──────────────────────────────────
def portfolio_variance(w):
    return w @ cov_train @ w

res_minvar = minimize(portfolio_variance, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-12})
w_minvar = res_minvar.x

# ── 3d. Maximum Diversification ───────────────────────────
# Maximise: (w · sigma_i) / sqrt(w' Σ w)
# where sigma_i = individual asset volatilities
asset_vols = np.sqrt(np.diag(cov_train))

def neg_diversification_ratio(w):
    weighted_vols = w @ asset_vols
    port_vol      = np.sqrt(w @ cov_train @ w + 1e-10)
    return -weighted_vols / port_vol

res_md = minimize(neg_diversification_ratio, w0, method="SLSQP",
                  bounds=bounds, constraints=constraints,
                  options={"maxiter": 1000, "ftol": 1e-12})
w_md = res_md.x

# ── 3e. Risk Parity (Equal Risk Contribution) ─────────────
# Each asset contributes equally to total portfolio variance
def risk_parity_objective(w):
    port_var = w @ cov_train @ w
    # marginal risk contributions
    mrc      = cov_train @ w
    # risk contributions
    rc       = w * mrc
    # minimise sum of squared differences between all pairs
    rc_mean  = port_var / NUM_ASSETS
    return np.sum((rc - rc_mean) ** 2)

res_rp = minimize(risk_parity_objective, w0, method="SLSQP",
                  bounds=bounds, constraints=constraints,
                  options={"maxiter": 2000, "ftol": 1e-14})
w_rp = res_rp.x / res_rp.x.sum()   # renormalize

# ── 3f. Inverse Volatility ────────────────────────────────
inv_vol  = 1.0 / (asset_vols + 1e-10)
w_invvol = inv_vol / inv_vol.sum()

# ── 3g. Momentum (rolling 20-day) ─────────────────────────
# Weights proportional to trailing 20-day return, rebalanced each period
# Applied to TEST set directly (walk-forward, no future leakage)
MOMENTUM_WINDOW = 20
# We need the raw returns just before the test period for the initial window
pre_test = returns_raw[train_end - MOMENTUM_WINDOW : train_end]   # last 20 train days

all_test = np.vstack([pre_test, test_returns])   # prepend warmup

mom_weights_list = []
for t in range(MOMENTUM_WINDOW, len(all_test)):
    window_ret = all_test[t - MOMENTUM_WINDOW:t]
    cumret     = (1 + window_ret).prod(axis=0) - 1   # cumulative return per asset
    scores     = np.maximum(cumret, 0)                # long-only: zero-weight losers
    total      = scores.sum()
    if total < 1e-10:
        mom_weights_list.append(np.ones(NUM_ASSETS) / NUM_ASSETS)
    else:
        mom_weights_list.append(scores / total)

mom_weights  = np.array(mom_weights_list)           # (T_test, N)
mom_port_ret = (test_returns * mom_weights).sum(axis=1)

print("\n── Classical Baseline Weights ───────────────────────")
for name, w in [("Equal Weight", w_ew), ("MV Max-Sharpe", w_mv),
                ("Min Variance", w_minvar), ("Max Diversification", w_md),
                ("Risk Parity", w_rp), ("Inv Volatility", w_invvol)]:
    print(f"  {name:22s}: {np.round(w, 3)}")

# =============================================================
# 4. MGAN MODEL
# =============================================================
from mgan import MGANPortfolio

sector_adj   = torch.load("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/static/A_sector.pt")
industry_adj = torch.load("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/static/A_industry.pt")
theme_adj    = torch.load("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/static/A_theme.pt")

print("\nsector  — shape:", sector_adj.shape,   "| nonzero rows:", (sector_adj.sum(1)   > 0).sum().item())
print("industry— shape:", industry_adj.shape, "| nonzero rows:", (industry_adj.sum(1) > 0).sum().item())
print("theme   — shape:", theme_adj.shape,    "| nonzero rows:", (theme_adj.sum(1)    > 0).sum().item())

model = MGANPortfolio(
    sector_adj=sector_adj,
    industry_adj=industry_adj,
    theme_adj=theme_adj,
    num_assets=NUM_ASSETS,
    hidden_dim=16,
    num_heads=4,
)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=5e-5)

num_epochs    = 40
best_val      = -np.inf
best_path     = "best_model.pt"
train_sharpes = []
val_sharpes   = []

for epoch in range(num_epochs):

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

    model.eval()
    with torch.no_grad():
        X_val_full, y_val_full = next(iter(val_loader))
        val_w      = model(X_val_full)
        val_port_r = (val_w * y_val_full).sum(dim=1)
        val_sharpe = annualized_sharpe(val_port_r).item()

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

# =============================================================
# 5. TEST EVALUATION
# =============================================================
print(f"\nBest Val Sharpe: {best_val:.4f}")
model.load_state_dict(torch.load(best_path))
model.eval()

with torch.no_grad():
    X_test_full, y_test_full = next(iter(test_loader))
    test_w      = model(X_test_full)
    test_port_r = (test_w * y_test_full).sum(dim=1)

# Collect all results
results = []

results.append(compute_all_metrics(test_port_r,                               "MGAN (ours)"))
results.append(compute_all_metrics(static_weights_to_tensor(w_ew,    y_test_full), "Equal Weight"))
results.append(compute_all_metrics(static_weights_to_tensor(w_mv,    y_test_full), "MV Max-Sharpe"))
results.append(compute_all_metrics(static_weights_to_tensor(w_minvar, y_test_full),"Min Variance"))
results.append(compute_all_metrics(static_weights_to_tensor(w_md,    y_test_full), "Max Diversification"))
results.append(compute_all_metrics(static_weights_to_tensor(w_rp,    y_test_full), "Risk Parity"))
results.append(compute_all_metrics(static_weights_to_tensor(w_invvol, y_test_full),"Inv Volatility"))
results.append(compute_all_metrics(
    torch.tensor(mom_port_ret, dtype=torch.float32), "Momentum (20d)"))

results_df = pd.DataFrame(results).set_index("Strategy")
print("\n" + "="*70)
print("FULL COMPARISON TABLE")
print("="*70)
print(results_df.to_string())
print("="*70)

# =============================================================
# 6. PLOTS
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Training curves ───────────────────────────────────────
axes[0].plot([-s for s in train_sharpes], label="Train Loss")
axes[0].plot([-s for s in val_sharpes],   label="Val Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (= −Sharpe)")
axes[0].set_title("Training vs Validation Loss")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(train_sharpes, label="Train Sharpe")
axes[1].plot(val_sharpes,   label="Val Sharpe")
axes[1].axhline(best_val,               color="green", linestyle="--", alpha=0.7,
                label=f"Best Val {best_val:.3f}")
eq_s = results_df.loc["Equal Weight", "Sharpe"]
axes[1].axhline(eq_s, color="red", linestyle="--", alpha=0.7,
                label=f"Equal Weight {eq_s:.3f}")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Annualized Sharpe")
axes[1].set_title("Training vs Validation Sharpe")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()

# ── Bar chart: Sharpe comparison ─────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))

colors = ["steelblue" if s != "MGAN (ours)" else "darkorange"
          for s in results_df.index]

axes2[0].bar(results_df.index, results_df["Sharpe"], color=colors)
axes2[0].axhline(0, color="black", linewidth=0.8)
axes2[0].set_title("Annualized Sharpe Ratio")
axes2[0].set_ylabel("Sharpe")
axes2[0].tick_params(axis="x", rotation=35)
axes2[0].grid(axis="y", alpha=0.4)

axes2[1].bar(results_df.index, results_df["CumRet"], color=colors)
axes2[1].axhline(0, color="black", linewidth=0.8)
axes2[1].set_title("Cumulative Return (Test Period)")
axes2[1].set_ylabel("Cumulative Return")
axes2[1].tick_params(axis="x", rotation=35)
axes2[1].grid(axis="y", alpha=0.4)

plt.tight_layout()
plt.savefig("baseline_comparison.png", dpi=150)
plt.show()

# ── Cumulative wealth curves ─────────────────────────────
fig3, ax3 = plt.subplots(figsize=(14, 6))

# MGAN
wealth_mgan = torch.cumprod(1 + test_port_r, dim=0).numpy()
ax3.plot(wealth_mgan, label="MGAN (ours)", linewidth=2, color="darkorange")

# All static baselines
static_baselines = [
    ("Equal Weight",        w_ew),
    ("MV Max-Sharpe",       w_mv),
    ("Min Variance",        w_minvar),
    ("Max Diversification", w_md),
    ("Risk Parity",         w_rp),
    ("Inv Volatility",      w_invvol),
]
linestyles = ["--", "-.", ":", "--", "-.", ":"]
for (name, w), ls in zip(static_baselines, linestyles):
    pr = test_returns @ w
    wealth = np.cumprod(1 + pr)
    ax3.plot(wealth, label=name, linestyle=ls, alpha=0.8)

# Momentum
wealth_mom = np.cumprod(1 + mom_port_ret)
ax3.plot(wealth_mom, label="Momentum (20d)", linestyle="--", color="purple", alpha=0.8)

ax3.axhline(1.0, color="black", linewidth=0.6, linestyle="--")
ax3.set_xlabel("Test Period (days)")
ax3.set_ylabel("Portfolio Value (starting at 1)")
ax3.set_title("Cumulative Wealth — All Strategies")
ax3.legend(loc="upper left", fontsize=8)
ax3.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("cumulative_wealth.png", dpi=150)
plt.show()