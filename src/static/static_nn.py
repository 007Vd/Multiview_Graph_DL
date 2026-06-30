#%%
import numpy as np 
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from mgan import MGANPortfolio
#%%
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
    y_train.append(train_returns[i])
x_train = np.array(x_train)
y_train = np.array(y_train)

x_test=[]
y_test=[]
for i in range(window_size,len(test_returns_std)):
    x_test.append(test_returns_std[i-window_size:i])
    y_test.append(test_returns[i])

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

train_loader = DataLoader(train_dataset, len(train_dataset), shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=len(test_dataset), shuffle=False)



sector_adj=torch.load("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/static/A_sector.pt")
industry_adj=torch.load("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/static/A_industry.pt")
theme_adj=torch.load("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/static/A_theme.pt")

print("sector  — shape:", sector_adj.shape,  "| nonzero rows:", (sector_adj.sum(1) > 0).sum().item())
print("industry— shape:", industry_adj.shape, "| nonzero rows:", (industry_adj.sum(1) > 0).sum().item())
print("theme   — shape:", theme_adj.shape,    "| nonzero rows:", (theme_adj.sum(1) > 0).sum().item())
print("sector density:",  sector_adj.mean().item())
print("industry density:", industry_adj.mean().item())
print("theme density:",    theme_adj.mean().item())
#%%
model = MGANPortfolio(
    sector_adj=sector_adj,
    industry_adj=industry_adj,
    theme_adj=theme_adj,
    num_assets=15,
    hidden_dim=16,
    num_heads=4
)

print(sum(p.numel() for p in model.parameters()))

def annualized_volatility(returns):
    return returns.std() * np.sqrt(252)


def sortino_ratio(returns):

    downside = returns[returns < 0]

    if len(downside) == 0:
        return torch.tensor(float("inf"))

    downside_std = downside.std()

    return (
        returns.mean()
        /
        (downside_std + 1e-8)
    ) * np.sqrt(252)


def max_drawdown(returns):

    wealth = torch.cumprod(
        1 + returns,
        dim=0
    )

    running_max = torch.cummax(
        wealth,
        dim=0
    )[0]

    drawdown = (
        wealth - running_max
    ) / running_max

    return drawdown.min()


def cumulative_return(returns):

    wealth = torch.cumprod(
        1 + returns,
        dim=0
    )

    return wealth[-1] - 1

# def sharpe_loss(weights, future_returns):
#     portfolio_returns = (weights * future_returns).sum(dim=1)
#     mean_return = portfolio_returns.mean()
#     std_return = portfolio_returns.std()
#     sharpe = mean_return / (std_return + 1e-8)
    
#     # Penalize uniform weights — push model to differentiate
#     weight_entropy = -(weights * (weights + 1e-8).log()).sum(dim=1).mean()
#     max_entropy = torch.log(torch.tensor(weights.shape[1], dtype=torch.float32))
#     diversity_penalty = weight_entropy / max_entropy  # normalized 0→1
    
#     return -sharpe + 0.1 * diversity_penalty  # penalize being too uniform

def sharpe_loss(weights,future_returns):
    portfolio_returns=(weights*future_returns).sum(dim=1)

    mean_return=portfolio_returns.mean()
    std_return=portfolio_returns.std()
    sharpe=mean_return/(std_return +1e-8)
    annualized_sharpe = sharpe

    return -annualized_sharpe

optimizer = torch.optim.Adam(model.parameters(), lr=5e-04, weight_decay=1e-05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=5e-05)

model.train()
X_batch, y_batch = next(iter(train_loader))
num_epochs = 150
train_sharpes = []
val_sharpes   = []
best_val      = -np.inf

for epoch in range(num_epochs):

    # ── Train ──────────────────────────
    model.train()
    optimizer.zero_grad()
    X_full, y_full = next(iter(train_loader))   # full dataset, one shot
    weights = model(X_full)
    loss = sharpe_loss(weights, y_full)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    # ── Eval ───────────────────────────
    model.eval()
    with torch.no_grad():
        train_port_r = (weights * y_full).sum(dim=1)
        train_sharpe = (train_port_r.mean() / (train_port_r.std() + 1e-8)) * np.sqrt(252)

        X_val, y_val = next(iter(test_loader))
        val_w        = model(X_val)
        val_port_r   = (val_w * y_val).sum(dim=1)
        val_sharpe   = (val_port_r.mean() / (val_port_r.std() + 1e-8)) * np.sqrt(252)

    if val_sharpe > best_val:
        best_val = val_sharpe

    train_sharpes.append(train_sharpe.item())
    val_sharpes.append(val_sharpe.item())

    print(f"Epoch {epoch+1}/{num_epochs} | Train: {train_sharpe:.4f} | Val: {val_sharpe:.4f} | Best Val: {best_val:.4f}")
    print(f"  Vol: {annualized_volatility(val_port_r):.4f} | Sortino: {sortino_ratio(val_port_r):.4f} | MaxDD: {max_drawdown(val_port_r):.4f} | CumRet: {cumulative_return(val_port_r):.4f}")

print(f"\nBest Val Sharpe reached: {best_val:.4f}")

model.eval()

with torch.no_grad():
    weights = model(x_test)

avg_weights = weights.mean(dim=0)

print(avg_weights)
print(avg_weights.sum())
print(weights.std(dim=0))

print("equal weights")

equal_weights = torch.ones(15) / 15

eq_returns = (y_test * equal_weights).sum(dim=1)

eq_sharpe = (eq_returns.mean()/(eq_returns.std() + 1e-8))*np.sqrt(252)
print(f"Equal Weights Annualized volatility: {annualized_volatility(eq_returns)}")
print(f"Equal Weights Sortino Ratio: {sortino_ratio(eq_returns)}")
print(f"Equal Weights Max DrawDown: {max_drawdown(eq_returns)}")
print(f"Equal Weights cumulative returns: {cumulative_return(eq_returns)}")

print("equal weight sharpe is :",eq_sharpe)

# plt.figure(figsize=(10,5))

# train_losses = [-x for x in train_sharpes]
# val_losses = [-x for x in val_sharpes]

# plt.plot(
#     train_losses,
#     label="Train Loss"
# )

# plt.plot(
#     val_losses,
#     label="Validation Loss"
# )

# plt.xlabel("Epoch")

# plt.ylabel("Sharpe Loss")

# plt.title("Training vs Validation Loss")

# plt.grid(True)

# plt.legend()

# plt.show()

# plt.figure(figsize=(10,5))

# plt.plot(
#     train_sharpes,
#     label="Train Sharpe"
# )

# plt.plot(
#     val_sharpes,
#     label="Validation Sharpe"
# )

# plt.xlabel("Epoch")

# plt.ylabel("Annualized Sharpe")

# plt.title("Training vs Validation Sharpe")

# plt.grid(True)

# plt.legend()

# plt.show()

# # %%
print(f"\nBest Val Sharpe reached: {best_val:.4f}")

# Equal weight baseline
equal_weights = torch.ones(15) / 15
eq_returns    = (y_val * equal_weights).sum(dim=1)
eq_sharpe     = (eq_returns.mean() / (eq_returns.std() + 1e-8)) * np.sqrt(252)
print(f"Equal Weight Sharpe: {eq_sharpe:.4f}")

# ── Plots ──────────────────────────────────────────────────
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
axes[1].axhline(best_val, color="green", linestyle="--",
                alpha=0.6, label=f"Best Val {best_val:.3f}")
axes[1].axhline(eq_sharpe.item(), color="red", linestyle="--",
                alpha=0.6, label=f"Equal Weight {eq_sharpe:.3f}")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Annualized Sharpe")
axes[1].set_title("Training vs Validation Sharpe")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()