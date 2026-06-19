#%%
import yfinance as yf
from pathlib import Path
import pandas as pd
RAW_DIR=Path("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/raw")
RAW_DIR.mkdir(parents=True,exist_ok=True)
# %%
START_DATE="2010-01-01"
END_DATE="2023-12-31"

DOW_30=["AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX",
    "DIS","GS","HD","HON","IBM","JNJ","JPM","KO",
    "MCD","MMM","MRK","MSFT","NKE","NVDA","PG","SHW",
    "TRV","UNH","V","VZ","WMT"]

stocks = [
    "AAPL",  # Tech
    "MSFT",  # Tech
    "NVDA",  # Tech/AI

    "JPM",   # Banking
    "V",     # Payments

    "JNJ",   # Healthcare
    "UNH",   # Healthcare

    "CVX",   # Energy

    "PG",    # Consumer Staples
    "WMT",   # Retail
    "HD",    # Consumer Discretionary

    "CAT",   # Industrials
    "HON",   # Industrials

    "DIS",   # Media
    "KO"     # Consumer Staples
]

# %%
for ticker in stocks:
    df=yf.download(ticker,
    start=START_DATE,
        end=END_DATE,
        progress=False,
        auto_adjust=False
)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    columns=['Date','Close','High', 'Low', 'Open', 'Volume']
    df=df[columns]
    df.columns.name = None
    SAVE_DIR=RAW_DIR/f"{ticker}.csv"

    print(f"SAVING {ticker} to {SAVE_DIR}")
    df.to_csv(SAVE_DIR)

# %%
files=list(Path(RAW_DIR).glob("*csv"))
print(f" the totoal nos of stocks used here are {len(files)}")

df=pd.read_csv(RAW_DIR/"AAPL.csv")
# print(df)

# %%
