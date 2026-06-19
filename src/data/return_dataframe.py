#%%
import pandas as pd
from pathlib import Path

RF=Path("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/return_dataframe")
RF.mkdir(parents=True,exist_ok=True)
PROCESSED_DIR=Path("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/processed")

stocks = [
    "AAPL","MSFT","NVDA","JPM","JNJ",
    "CVX","PG","CAT","V","WMT",
    "UNH","HON","HD","DIS","KO"
]
return_df=pd.DataFrame()

for stock in stocks:
    df=pd.read_csv(PROCESSED_DIR/f"{stock}.csv")
    if "Date" not in return_df.columns:
        return_df["Date"] = df["Date"]

    return_df[stock]=df["Return"]
    

return_df=return_df.dropna()

return_df.to_csv(RF/f"return_df")
print("return df has been saved")