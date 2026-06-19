#%%
from pathlib import Path
import pandas as pd

RAW_DIR=Path("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/raw")
all_dates=None
for file in RAW_DIR.glob("*csv"):
    df=pd.read_csv(file)
    dates=pd.to_datetime(df["Date"])
    if all_dates is None:
        all_dates=set(dates)

    else:
        all_dates=all_dates.intersection(set(dates))

calendar = sorted(list(all_dates))
calendar = pd.DatetimeIndex(calendar)
print(len(calendar))
# %%
PROCESSED_DIR=Path("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/processed")
PROCESSED_DIR.mkdir(parents=True,exist_ok=True)

for file in RAW_DIR.glob("*csv"):
    df=pd.read_csv(file)
    df["Date"] = pd.to_datetime(df["Date"])
    df = (df.set_index("Date").reindex(calendar))
    df.index.name = "Date"
    
    df = df.drop(columns=["Unnamed: 0"])
    df["Return"] = df["Close"].pct_change()
    df=df.dropna()
    print(f"{file.name} processing completed...")

    df.to_csv(PROCESSED_DIR/file.name)

#%%
for file in PROCESSED_DIR.glob("*.csv"):

    df = pd.read_csv(file)

    print(
        file.stem,
        df.isna().sum().sum()
    )