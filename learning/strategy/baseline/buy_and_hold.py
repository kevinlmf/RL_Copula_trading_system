import pandas as pd
import numpy as np
import os

df = pd.read_csv("data/raw_data/real_asset_log_returns_extreme.csv", index_col=0)
returns = df.values

# ✅ 假设买入第一只资产并持有
returns_list = returns[:, 0].tolist()

os.makedirs("results", exist_ok=True)
pd.DataFrame({"Returns": returns_list}).to_csv("results/buy_and_hold_returns.csv", index=False)
print("✅ Saved buy_and_hold_returns.csv")
