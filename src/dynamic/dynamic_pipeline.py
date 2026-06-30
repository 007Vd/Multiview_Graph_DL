import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import minimize
from dynamic_graph import DynamicGraphBuilder
# ── Reproducibility ───────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
# =============================================================
# 1. DATA & DATASET
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
class DynamicGraphDataset(Dataset):
    def __init__(self, features, labels, momentum_window=20, top_k=3):
        self.features = features
        self.labels   = labels
        self.graph_builder = DynamicGraphBuilder(momentum_window=momentum_window, top_k=top_k)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        X_window = self.features[idx]
        # Build dynamic adjacency strictly from the input window
        dynamic_adj = self.graph_builder.build_graph(X_window)
        return X_window, dynamic_adj, self.labels[idx]
train_loader = DataLoader(DynamicGraphDataset(x_train, y_train), batch_size=len(y_train), shuffle=True)
val_loader   = DataLoader(DynamicGraphDataset(x_val,   y_val),   batch_size=len(y_val),   shuffle=False)
test_loader  = DataLoader(DynamicGraphDataset(x_test,  y_test),  batch_size=len(y_test),  shuffle=False)
# =============================================================
# 2. METRICS & CLASSICAL BASELINES (Copied from your script)
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
def static_weights_to_tensor(w_np, test_ret_tensor):
    w = torch.tensor(w_np, dtype=torch.float32)
    return (test_ret_tensor * w).sum(dim=1)
w_ew = np.ones(NUM_ASSETS) / NUM_ASSETS
mu_train  = train_returns.mean(axis=0)
cov_train = np.cov(train_returns.T)
def neg_sharpe_np(w):
    port_r = mu_train @ w
    port_v = np.sqrt(w @ cov_train @ w + 1e-10)
    return -port_r / port_v
constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
bounds      = [(0, 1)] * NUM_ASSETS
w0          = np.ones(NUM_ASSETS) / NUM_ASSETS
w_mv = minimize(neg_sharpe_np, w0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 1000, "ftol": 1e-12}).x
w_minvar = minimize(lambda w: w @ cov_train @ w, w0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 1000, "ftol": 1e-12}).x
asset_vols = np.sqrt(np.diag(cov_train))
def neg_diversification_ratio(w):
    weighted_vols = w @ asset_vols
    port_vol      = np.sqrt(w @ cov_train @ w + 1e-10)
    return -weighted_vols / port_vol
w_md = minimize(neg_diversification_ratio, w0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 1000, "ftol": 1e-12}).x
def risk_parity_objective(w):
    port_var = w @ cov_train @ w
    rc = w * (cov_train @ w)
    rc_mean = port_var / NUM_ASSETS
    return np.sum((rc - rc_mean) ** 2)
res_rp = minimize(risk_parity_objective, w0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 2000, "ftol": 1e-14})
w_rp = res_rp.x / res_rp.x.sum()
inv_vol  = 1.0 / (asset_vols + 1e-10)
w_invvol = inv_vol / inv_vol.sum()
MOMENTUM_WINDOW = 20
pre_test = returns_raw[train_end - MOMENTUM_WINDOW : train_end]
all_test = np.vstack([pre_test, test_returns])
mom_weights_list = []
for t in range(MOMENTUM_WINDOW, len(all_test)):
    window_ret = all_test[t - MOMENTUM_WINDOW:t]
    cumret     = (1 + window_ret).prod(axis=0) - 1
    scores     = np.maximum(cumret, 0)
    total      = scores.sum()
    if total < 1e-10:
        mom_weights_list.append(np.ones(NUM_ASSETS) / NUM_ASSETS)
    else:
        mom_weights_list.append(scores / total)
mom_weights  = np.array(mom_weights_list)
mom_port_ret = (test_returns * mom_weights).sum(axis=1)
# =============================================================
# 3. DYNAMIC MGAN MODEL
# =============================================================
class HistoricalState(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super().__init__()
        self.W1 = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.Ve = nn.Parameter(torch.randn(hidden_dim)* 0.1)
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
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        B = x.shape[0]
        x = x.permute(0, 2, 1)
        x = x.reshape(B * self.num_assets, -1, 1)
        H, _ = self.lstm(x) 
        X_last = x[:, -1, :]
        E, alpha = self.attn(H, X_last) 
        E = E.view(B, self.num_assets, self.hidden_dim)
        return self.norm(E), alpha
class DynamicTGAHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W        = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.a        = nn.Linear(2 * hidden_dim, 1, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
    def forward(self, E, dynamic_adj):
        B, N, D = E.shape
        H = self.W(E)
        Hv = H.unsqueeze(2).expand(-1, -1, N, -1)
        Hu = H.unsqueeze(1).expand(-1, N, -1, -1)
        scores = self.a(torch.cat([Hu, Hv], dim=-1)).squeeze(-1)
        
        # Soft mask: multiply by dynamic_adj
        scores = scores * dynamic_adj
        
        alpha  = F.softmax(scores, dim=2)
        output = torch.bmm(alpha, H)
        output = self.out_proj(output)
        return output, alpha
class DynamicMGANLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.heads = nn.ModuleList([DynamicTGAHead(hidden_dim) for _ in range(num_heads)])
        self.fusion = nn.Linear(num_heads*hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, E, dynamic_adj):
        outputs = []
        attentions = []
        for head in self.heads:
            out, alpha = head(E, dynamic_adj)
            outputs.append(out)
            attentions.append(alpha)
        multihead = torch.cat(outputs, dim=-1)
        embeddings = self.fusion(multihead)
        return self.norm(embeddings + E), attentions
class PortfolioHead(nn.Module):
    def __init__(self, hidden_dim, num_graphs=1):
        super().__init__()
        # E_stock + number of graphs * E_graph
        in_dim = hidden_dim * (1 + num_graphs)
        self.norm = nn.LayerNorm(in_dim)
        self.network = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        x = self.norm(x)    
        scores = self.network(x).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        return weights
class DynamicMGANPortfolio(nn.Module):
    def __init__(self, num_assets=15, hidden_dim=64, num_heads=4):
        super().__init__()
        self.encoder = StockEncoder(num_assets=num_assets, hidden_dim=hidden_dim)
        self.dynamic_graph = DynamicMGANLayer(hidden_dim=hidden_dim, num_heads=num_heads)
        # Using 1 graph (Dynamic)
        self.portfolio_head = PortfolioHead(hidden_dim=hidden_dim, num_graphs=1)
    def forward(self, x, dynamic_adj):
        E_stock, stock_attention = self.encoder(x)
        E_dynamic, dynamic_attention = self.dynamic_graph(E_stock, dynamic_adj)
        fused_embeddings = torch.cat([E_stock, E_dynamic], dim=-1)
        weights = self.portfolio_head(fused_embeddings)
        return weights
# =============================================================
# 4. TRAINING
# =============================================================
model = DynamicMGANPortfolio(num_assets=NUM_ASSETS, hidden_dim=16, num_heads=4)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=5e-5)
num_epochs    = 205
best_val      = -np.inf
best_path     = "/Users/007vd/Downloads/DAU/Split_Graph_dl/src/dynamic/best_dynamic_model.pt"
train_sharpes = []
val_sharpes   = []
for epoch in range(num_epochs):
    model.train()
    batch_sharpes = []
    for X_batch, adj_batch, y_batch in train_loader:
        optimizer.zero_grad()
        weights = model(X_batch, adj_batch)
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
        X_val_full, adj_val_full, y_val_full = next(iter(val_loader))
        val_w      = model(X_val_full, adj_val_full)
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
    X_test_full, adj_test_full, y_test_full = next(iter(test_loader))
    test_w      = model(X_test_full, adj_test_full)
    test_port_r = (test_w * y_test_full).sum(dim=1)
results = []
results.append(compute_all_metrics(test_port_r,                               "Dynamic MGAN (ours)"))
results.append(compute_all_metrics(static_weights_to_tensor(w_ew,    y_test_full), "Equal Weight"))
results.append(compute_all_metrics(static_weights_to_tensor(w_mv,    y_test_full), "MV Max-Sharpe"))
results.append(compute_all_metrics(static_weights_to_tensor(w_minvar, y_test_full),"Min Variance"))
results.append(compute_all_metrics(static_weights_to_tensor(w_md,    y_test_full), "Max Diversification"))
results.append(compute_all_metrics(static_weights_to_tensor(w_rp,    y_test_full), "Risk Parity"))
results.append(compute_all_metrics(static_weights_to_tensor(w_invvol, y_test_full),"Inv Volatility"))
results.append(compute_all_metrics(torch.tensor(mom_port_ret, dtype=torch.float32), "Momentum (20d)"))
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
axes[0].plot([-s for s in train_sharpes], label="Train Loss")
axes[0].plot([-s for s in val_sharpes],   label="Val Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (= −Sharpe)")
axes[0].set_title("Training vs Validation Loss")
axes[0].grid(True)
axes[0].legend()
axes[1].plot(train_sharpes, label="Train Sharpe")
axes[1].plot(val_sharpes,   label="Val Sharpe")
axes[1].axhline(best_val, color="green", linestyle="--", alpha=0.7, label=f"Best Val {best_val:.3f}")
eq_s = results_df.loc["Equal Weight", "Sharpe"]
axes[1].axhline(eq_s, color="red", linestyle="--", alpha=0.7, label=f"Equal Weight {eq_s:.3f}")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Annualized Sharpe")
axes[1].set_title("Training vs Validation Sharpe")
axes[1].grid(True)
axes[1].legend()
plt.tight_layout()
plt.savefig("/Users/007vd/Downloads/DAU/Split_Graph_dl/src/dynamic/training_curves_dynamic.png", dpi=150)
plt.close()
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))
colors = ["steelblue" if s != "Dynamic MGAN (ours)" else "darkorange" for s in results_df.index]
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
plt.savefig("/Users/007vd/Downloads/DAU/Split_Graph_dl/src/dynamic/baseline_comparison_dynamic.png", dpi=150)
plt.close()
fig3, ax3 = plt.subplots(figsize=(14, 6))
wealth_mgan = torch.cumprod(1 + test_port_r, dim=0).numpy()
ax3.plot(wealth_mgan, label="Dynamic MGAN (ours)", linewidth=2, color="darkorange")
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
wealth_mom = np.cumprod(1 + mom_port_ret)
ax3.plot(wealth_mom, label="Momentum (20d)", linestyle="--", color="purple", alpha=0.8)
ax3.axhline(1.0, color="black", linewidth=0.6, linestyle="--")
ax3.set_xlabel("Test Period (days)")
ax3.set_ylabel("Portfolio Value (starting at 1)")
ax3.set_title("Cumulative Wealth — All Strategies")
ax3.legend(loc="upper left", fontsize=8)
ax3.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("/Users/007vd/Downloads/DAU/Split_Graph_dl/src/dynamic/cumulative_wealth_dynamic.png", dpi=150)
plt.close()
