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

#%%
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
split=int(0.8*len(X))
X_train = X[:split]
y_train =y[:split]

X_test = X[split:]
y_test = y[split:]

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(y_test, dtype=torch.float32)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

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
class HistoricalState(nn.Module):
    def __init__(self,hidden_dim,input_dim):
        super().__init__()
        self.W1=nn.Linear(2*hidden_dim,hidden_dim,bias=False)
        self.W2=nn.Linear(input_dim,hidden_dim,bias=False)
        self.Ve=nn.Parameter(torch.randn(hidden_dim))

    def forward(self,H,X_last):
        B,L,Hdim=H.shape
        H_last=H[:,-1,:]
        H_last=H_last.unsqueeze(1)
        H_last=H_last.repeat(1,L,1)
        concat=torch.cat([H,H_last],dim=-1)
        term1=self.W1(concat)
        term2=self.W2(X_last).unsqueeze(1)
        e=torch.tanh(term1+term2)

        scores=torch.matmul(e,self.Ve)

        alpha=torch.softmax(scores,dim=1)
        E=(alpha.unsqueeze(-1)*H).sum(dim=1)

        return E,alpha
    
class PortfolioLSTM(nn.Module):
    def __init__(self,num_assets=15,hidden_dim=256):
        super().__init__()
        self.lstm=nn.LSTM(
            input_size=num_assets,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.attn=HistoricalState(hidden_dim,num_assets)
        self.mlp=nn.Sequential(
            nn.Linear(hidden_dim,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,num_assets)
            )
    
    def forward(self,x):
        # B = x.shape[0]
        # x = x.permute(0,2,1)
        # x = x.reshape(B*15,30,1)
        H,_=self.lstm(x)
        X_last=x[:,-1,:]
        E,alpha=self.attn(H,X_last)
        logits=self.mlp(E)
        weights=F.softmax(logits,dim=1)
        return weights
# %%
model=PortfolioLSTM(X_train.shape[2])
print(sum(p.numel() for p in model.parameters()))
print(model)
# %%
def sharpe_loss(weights,future_returns):
    portfolio_returns=(weights*future_returns).sum(dim=1)

    mean_return=portfolio_returns.mean()
    std_return=portfolio_returns.std()
    sharpe=mean_return/(std_return +1e-8)
    annualized_sharpe = sharpe

    return -annualized_sharpe

optimizer=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-4)
# %%
def annualized_volatility(returns):
    return returns.std() * np.sqrt(252)

def sortino_ratio(returns):
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0:
        return torch.tensor(float("inf"))

    downside_std = downside_returns.std()

    return (returns.mean()/(downside_std + 1e-8)) * np.sqrt(252)

def max_drawdown(returns):
    wealth = torch.cumprod(1 + returns,dim=0)
    running_max = torch.cummax(wealth,dim=0)[0]
    drawdown = (wealth - running_max) / running_max

    return drawdown.min()

def cumulative_return(returns):
    wealth = torch.cumprod(1 + returns,dim=0)
    return wealth[-1] - 1

# %%
train_sharpes = []
val_sharpes = []

num_epochs = 24

for epoch in range(num_epochs):
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

    
    model.eval()

    with torch.no_grad():

        
        train_returns = []

        for X_batch, y_batch in train_loader:

            weights = model(X_batch)
            

            portfolio_returns = (weights * y_batch).sum(dim=1)

            train_returns.append(portfolio_returns)

        train_returns = torch.cat(train_returns,dim=0)
    
        train_sharpe = (train_returns.mean()/(train_returns.std() + 1e-8))
        train_sharpe=train_sharpe*np.sqrt(252)
        train_sharpes.append(
            train_sharpe.item()
        )

        val_returns = []

        for X_batch, y_batch in test_loader:

            weights = model(X_batch)

            portfolio_returns = (weights * y_batch).sum(dim=1)

            val_returns.append(portfolio_returns)

        val_returns = torch.cat(val_returns,dim=0)

        val_sharpe = (val_returns.mean()/(val_returns.std() + 1e-8)) *np.sqrt(252)

        val_sharpes.append(val_sharpe.item())
       

    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"Train Sharpe: {train_sharpe:.4f} | "
        f"Val Sharpe: {val_sharpe:.4f}"
        
    )
    print(f"Attention LSTM Annualized volatility: {annualized_volatility(val_returns)}")
    print(f"Attention LSTM Sortino Ratio: {sortino_ratio(val_returns)}")
    print(f"Attention LSTM Max DrawDown: {max_drawdown(val_returns)}")
    print(f"Attention LSTM cumulative returns: {cumulative_return(val_returns)}")


# %%
print("equal weights")

equal_weights = torch.ones(15) / 15

eq_returns = (Y_test * equal_weights).sum(dim=1)

eq_sharpe = (eq_returns.mean()/(eq_returns.std() + 1e-8))*np.sqrt(252)
print(f"Equal Weights Annualized volatility: {annualized_volatility(eq_returns)}")
print(f"Equal Weights Sortino Ratio: {sortino_ratio(eq_returns)}")
print(f"Equal Weights Max DrawDown: {max_drawdown(eq_returns)}")
print(f"Equal Weights cumulative returns: {cumulative_return(eq_returns)}")

print("equal weight sharpe is :",eq_sharpe)

import matplotlib.pyplot as plt

train_losses = [-x for x in train_sharpes]
val_losses = [-x for x in val_sharpes]

fig, ax = plt.subplots(
    1,
    2,
    figsize=(14,5)
)

ax[0].plot(train_losses,label="Train Loss")

ax[0].plot(val_losses,label="Validation Loss")

ax[0].set_title("Loss vs Epoch")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Los`s")

ax[0].legend()
ax[0].grid(True)

ax[1].plot(train_sharpes,label="Train Sharpe")

ax[1].plot(val_sharpes,label="Validation Sharpe")

ax[1].set_title("Sharpe vs Epoch")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Sharpe")

ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()


# %%
