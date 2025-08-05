import pandas as pd
import numpy as np
import os

df = pd.read_csv("data/raw_data/real_asset_log_returns_extreme.csv", index_col=0)
returns = df.values

# ✅ 均衡权重，直接使用收益率点乘
weights = np.ones(returns.shape[1]) / returns.shape[1]
returns_list = [np.dot(weights, r) for r in returns]  # 百分比收益率

os.makedirs("results", exist_ok=True)
pd.DataFrame({"Returns": returns_list}).to_csv("results/equal_weight_returns.csv", index=False)
print("✅ Saved equal_weight_returns.csv")



