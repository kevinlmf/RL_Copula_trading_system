import os
import pandas as pd
import matplotlib.pyplot as plt

# === 参数配置 ===
RESULT_DIR = "results"
STRATEGY_FILES = {
    "CPPO": "cppo_returns.csv",
    "SAC": "sac_returns.csv",
    "Equal Weight": "equal_weight_returns.csv",
    "Buy & Hold": "buy_and_hold_returns.csv",
    "Copula CVaR Opt": "copula_cvar_returns.csv"
}
PLOT_PATH = os.path.join(RESULT_DIR, "strategy_comparison.png")
CSV_OUT = os.path.join(RESULT_DIR, "strategy_metrics_summary.csv")

# ✅ 修正版 calculate_metrics
def calculate_metrics(returns: pd.Series, freq=252):
    daily_returns = returns.dropna()  # ✅ 本身就是 daily return，无需 pct_change()
    avg_daily = daily_returns.mean()
    std_daily = daily_returns.std()

    if std_daily == 0 or pd.isna(std_daily):
        sharpe_ratio = float("nan")
    else:
        sharpe_ratio = avg_daily / std_daily * (freq ** 0.5)

    # 防止爆炸，限制复利增长计算
    if avg_daily > -1:
        annual_return = (1 + avg_daily) ** freq - 1
    else:
        annual_return = -1  # 最坏情况处理

    # ✅ 正确计算累计收益 & 最大回撤
    cum_returns = (1 + daily_returns).cumprod()
    max_drawdown = ((cum_returns.cummax() - cum_returns) / cum_returns.cummax()).max()

    return round(annual_return * 100, 2), round(sharpe_ratio, 2), round(-max_drawdown * 100, 2)

def main():
    metrics = []
    plt.figure(figsize=(12, 6))

    for name, filename in STRATEGY_FILES.items():
        path = os.path.join(RESULT_DIR, filename)
        if not os.path.exists(path):
            print(f"[❌ Missing] {path}")
            continue

        df = pd.read_csv(path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
        elif df.columns[0] != "Returns":
            df.columns = ["Returns"]

        returns = df.iloc[:, 0]

        # ✅ 正确绘制累计收益率曲线（直接用 return）
        cum_return = (1 + returns.fillna(0)).cumprod()
        plt.plot(cum_return.values, label=name)

        # === 计算指标 ===
        annual_ret, sharpe, max_dd = calculate_metrics(returns)
        metrics.append({
            "Strategy": name,
            "Annual Return (%)": annual_ret,
            "Sharpe Ratio": sharpe,
            "Max Drawdown (%)": max_dd
        })

    # === 输出图表与表格 ===
    plt.title("Strategy Comparison: Cumulative Return")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"[📈 Saved] {PLOT_PATH}")

    metrics_df = pd.DataFrame(metrics)
    metrics_df.sort_values("Sharpe Ratio", ascending=False, inplace=True)
    metrics_df.to_csv(CSV_OUT, index=False)
    print(f"[📄 Saved] {CSV_OUT}")
    print(metrics_df)

if __name__ == "__main__":
    main()

