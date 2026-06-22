from openai import OpenAI
from dotenv import load_dotenv
import yfinance as yf
import json
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY")
)

tickers = [
    "AAPL",
    "MSFT",
    "NVDA",
    "JPM",
    "V",
    "JNJ",
    "UNH",
    "CVX",
    "PG",
    "WMT",
    "HD",
    "CAT",
    "HON",
    "DIS",
    "KO"
]

metadata = []

print("Fetching sector and industry data from yfinance...")

for ticker in tickers:
    try:
        info = yf.Ticker(ticker).info

        metadata.append({
            "ticker": ticker,
            "company": info.get("longName", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", "")
        })

        print(f"Fetched {ticker}")

    except Exception as e:
        print(f"Failed {ticker}: {e}")

print("\nGenerating themes using GPT...\n")

prompt = f"""
For each stock below, generate exactly 5 investment themes.

Return ONLY valid JSON.

Format:

[
    {{
        "ticker":"AAPL",
        "themes":[
            "Theme1",
            "Theme2",
            "Theme3",
            "Theme4",
            "Theme5"
        ]
    }}
]

Stocks:

{json.dumps(metadata, indent=2)}
"""

response = client.responses.create(
    model="gpt-5-mini",
    input=prompt
)

theme_data = json.loads(response.output_text)

theme_map = {
    item["ticker"]: item["themes"]
    for item in theme_data
}

for stock in metadata:
    stock["themes"] = theme_map.get(
        stock["ticker"],
        []
    )

output_path = "/Users/007vd/Downloads/DAU/Split_Graph_dl/data/stock_metadata.json"

with open(output_path, "w") as f:
    json.dump(
        metadata,
        f,
        indent=4
    )

print(f"\nSaved metadata for {len(metadata)} stocks")
print(f"Output: {output_path}")