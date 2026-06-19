#%%
import numpy as np
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
DF=Path("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/simple_mlp/return_dataframe/return_df")

returns_df=pd.read_csv(DF)
print(returns_df.columns)
returns_df=returns_df.drop(columns=["Date",'Unnamed: 0'])
print(returns_df)

# %%
returns=returns_df.values
X=[]
y=[]
window=30
for i in range(window,len(returns)):
    X.append(returns[i-window:i])
    y.append(returns[i])

X=np.array(X)
y=np.array(y)


print(X.shape)
print(y.shape)

# %%
X=X.reshape(X.shape[0],-1)
X.shape
# %%
split=int(0.8*len(X))
X_train = X[:split]
y_train =y[:split]

X_test = X[split:]
y_test = y[split:]


X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(y_test, dtype=torch.float32)
# %%
class CustomDataset(Dataset):
    def __init__(self,features,labels):
        self.features=features
        self.labels=labels

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self,index):
        return self.features[index] , self.labels[index]
    
train_dataset=CustomDataset(X_train,y_train)
test_dataset=CustomDataset(X_test,y_test)
# %%
train_loader=DataLoader(train_dataset,batch_size=len(train_dataset),shuffle=False,pin_memory=True)
test_loader=DataLoader(test_dataset,batch_size=len(test_dataset),shuffle=False,pin_memory=True)
# %%
class PortfolioMLP(nn.Module):
    def __init__(self,num_features):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(num_features,256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout(p=0.3),
            nn.Linear(256,128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(p=0.3),
            nn.Linear(128,15)
        )

    def forward(self,x):
        logits=self.net(x)
        weights=F.softmax(logits,dim=1)

        return weights

# %%
model=PortfolioMLP(X_train.shape[1])
sum(p.numel() for p in model.parameters())
model
# %%
def sharpe_loss(weights,future_returns):
    portfolio_returns=(weights*future_returns).sum(dim=1)

    mean_return=portfolio_returns.mean()
    std_return=portfolio_returns.std()
    sharpe=mean_return/(std_return +1e-8)
    annualized_sharpe = sharpe

    return -annualized_sharpe

optimizer=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-4)

#%%
train_sharpes = []
val_sharpes = []

num_epochs = 50

for epoch in range(num_epochs):

    # =========================
    # Training
    # =========================
    model.train()

    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()

        weights = model(X_batch)

        loss = sharpe_loss(
            weights,
            y_batch
        )

        loss.backward()

        optimizer.step()

    # =========================
    # Evaluation
    # =========================
    model.eval()

    with torch.no_grad():

        # ----- Train Sharpe -----
        train_returns = []

        for X_batch, y_batch in train_loader:

            weights = model(X_batch)
            

            portfolio_returns = (
                weights * y_batch
            ).sum(dim=1)

            train_returns.append(
                portfolio_returns
            )

        train_returns = torch.cat(
            train_returns,
            dim=0
        )
        

        train_sharpe = (
            train_returns.mean()
            /
            (train_returns.std() + 1e-8)
        )
        train_sharpe=train_sharpe*np.sqrt(252)
        train_sharpes.append(
            train_sharpe.item()
        )

        # ----- Validation Sharpe -----
        val_returns = []

        for X_batch, y_batch in test_loader:

            weights = model(X_batch)

            portfolio_returns = (
                weights * y_batch
            ).sum(dim=1)

            val_returns.append(
                portfolio_returns
            )

        val_returns = torch.cat(
            val_returns,
            dim=0
        )

        val_sharpe = (
            val_returns.mean()
            /
            (val_returns.std() + 1e-8)
        ) *np.sqrt(252)

        val_sharpes.append(
            val_sharpe.item()
        )

    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"Train Sharpe: {train_sharpe:.4f} | "
        f"Val Sharpe: {val_sharpe:.4f}"
    )


#%%
model.eval()

with torch.no_grad():
    weights = model(X_test)

avg_weights = weights.mean(dim=0)

print(avg_weights)
print(avg_weights.sum())
print(weights.std(dim=0))

print("equal wefights")

equal_weights = torch.ones(15) / 15

eq_returns = (
    Y_test * equal_weights
).sum(dim=1)

eq_sharpe = (
    eq_returns.mean()
    /
    (eq_returns.std() + 1e-8)
)*np.sqrt(252)

print("equal eright sharpe is :",eq_sharpe)

#%%
import matplotlib.pyplot as plt

train_losses = [-x for x in train_sharpes]
val_losses = [-x for x in val_sharpes]

fig, ax = plt.subplots(
    1,
    2,
    figsize=(14,5)
)

# =========================
# Loss Curve
# =========================
ax[0].plot(
    train_losses,
    label="Train Loss"
)

ax[0].plot(
    val_losses,
    label="Validation Loss"
)

ax[0].set_title("Loss vs Epoch")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")

ax[0].legend()
ax[0].grid(True)

# =========================
# Sharpe Curve
# =========================
ax[1].plot(
    train_sharpes,
    label="Train Sharpe"
)

ax[1].plot(
    val_sharpes,
    label="Validation Sharpe"
)

ax[1].set_title("Sharpe vs Epoch")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Sharpe")

ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()


# %%
# 

#%%
# import optuna

# study = optuna.create_study(
#     direction="maximize"
# )

# study.optimize(
#     objective,
#     n_trials=50
# )

# print(
#     "Best Sharpe:",
#     study.best_value
# )

# print(
#     study.best_params
# )


# %%
# equal_weights = torch.ones(15) / 15

# portfolio_returns = (
#     Y_test * equal_weights
# ).sum(dim=1)

# equal_weight_sharpe = (
#     portfolio_returns.mean()
#     /
#     (portfolio_returns.std() + 1e-8)
# )

# print(equal_weight_sharpe*np.sqrt(252))
# %%
