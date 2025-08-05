import pandas as pd
import numpy as np
import os
from learning.copula.gaussian_copula import GaussianCopula

df = pd.read_csv("data/raw_data/real_asset_log_returns_extreme.csv", index_col=0)
returns = df.values

# ✅ 拟合 Gaussian Copula 并估计风险
copula = GaussianCopula()
copula.fit(returns)

mean_returns = returns.mean(axis=0)
quantile = np.quantile(returns, 0.05, axis=0)
cvar = returns[returns <= quantile].mean(axis=0)  # 粗略估计 CVaR
lam = 10
weights = mean_returns - lam * cvar
weights = np.clip(weights, 0, None)
weights /= weights.sum()

# ✅ 使用百分比收益率点乘
returns_list = [np.dot(weights, r) for r in returns]

os.makedirs("results", exist_ok=True)
pd.DataFrame({"Returns": returns_list}).to_csv("results/copula_cvar_returns.csv", index=False)
print("✅ Saved copula_cvar_returns.csv")



