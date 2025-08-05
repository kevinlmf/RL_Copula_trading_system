import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os

# === 1. Use assets with complete 2020 data ===
tickers = [
    "SPY", "QQQ", "DIA", "IWM", "VTI",       # US Equity Indices
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", # Big Tech
    "JPM", "BAC", "XOM", "CVX", "GE",        # Banks + Industrials
    "TLT", "IEF", "LQD",                     # Bonds
    "GLD", "SLV", "USO", "DBC",              # Commodities
    "XLF", "XLK", "XLE", "XLV", "XLY",       # SPDR Sector ETFs
    "XLI", "XLU"                             # Industrials, Utilities
]

# === 2. Download only 2020 (COVID crash) ===
print("📥 Downloading 2020 COVID crash data...")
data = yf.download(tickers, start="2020-01-01", end="2020-12-31", auto_adjust=True)

# Extract closing prices
close_data = data['Close']

# === 3. Compute daily log returns ===
log_returns = np.log(close_data / close_data.shift(1)).dropna()

# Drop assets with any missing data
log_returns = log_returns.dropna(axis=1, how="any")
log_returns.columns = [f"asset_{i+1}" for i in range(len(log_returns.columns))]

print(f"✅ Kept {log_returns.shape[1]} assets after filtering.")
print("✅ Preview of 2020 COVID daily log returns:")
print(log_returns.head())

# === 4. Save to CSV ===
save_path = "data/mid_dimension/real_asset_log_returns_extreme.csv"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
log_returns.to_csv(save_path, index=True)
print(f"✅ Saved 2020 COVID log returns to: {save_path}")

# === 5. Visualize Spearman correlation ===
def plot_spearman(df):
    corr, _ = spearmanr(df)
    corr_matrix = pd.DataFrame(corr, index=df.columns, columns=df.columns)

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", square=True)
    plt.title("📘 Spearman Correlation — COVID Crash (2020)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

plot_spearman(log_returns)


