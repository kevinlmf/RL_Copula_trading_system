
# RL_Copula_trading_system

A hybrid algorithmic trading system integrating **Reinforcement Learning (RL)** with **Copula-based risk modeling**. This project benchmarks multiple trading strategies—including SAC, CPPO, Equal Weight, Buy-and-Hold, and Copula-CVaR optimization—under real-world financial data, and provides cumulative return plots and strategy performance metrics.

## 📌 Features

- ✅ Deep RL agents: Soft Actor-Critic (SAC), Constrained PPO (CPPO)
- ✅ Copula-based CVaR risk optimization using Gaussian Copula
- ✅ Baseline strategies: Equal Weight, Buy & Hold
- ✅ Performance metrics: Annual Return, Sharpe Ratio, Max Drawdown
- ✅ Full backtesting & visualization pipeline

## 📁 Project Structure

```
RL_Copula_trading_system/
├── learning/                # Core logic (env, agent, copula, strategy)
├── results/                 # Strategy returns & plots
├── scripts/                 # Training scripts & strategy runners
├── compare_strategies.py    # Evaluation & visualization
└── run_all_strategies.py    # One-click strategy execution
```

## 🚀 How to Run

1. **Install dependencies** (Python ≥ 3.8):

```bash
pip install -r requirements.txt
```

2. **Run all strategies:**

```bash
export PYTHONPATH=.
python run_all_strategies.py
```

3. **Compare strategies:**

```bash
python compare_strategies.py
```

4. **Output:**

- 📈 `results/strategy_comparison.png`
- 📄 `results/strategy_metrics_summary.csv`

## 📊 Example Metrics (Auto-generated)

| Strategy         | Annual Return (%) | Sharpe Ratio | Max Drawdown (%) |
|------------------|-------------------|--------------|------------------|
| Buy & Hold       | 79.92             | 1.26         | -34.34           |
| SAC              | 18.18             | 0.62         | -38.53           |
| Equal Weight     | 13.83             | 0.41         | -36.21           |
| Copula CVaR Opt  | 14.00             | 0.41         | -36.20           |
| CPPO             | 0.44              | 0.01         | -74.83           |

## 🧠 Background

- **Copula models** capture joint dependence structures beyond correlation.
- **CVaR optimization** targets extreme risk scenarios.
- **RL agents** learn to optimize portfolio allocation under uncertainty.

## 📈 Future Work

- ⚡ Train RL agents using GPU acceleration for speed and scalability
- 🤖 Add more advanced RL algorithms (e.g., TD3, PPO-Lagrangian, Offline RL)
- 💹 Extend to more financial environments such as **high-frequency trading (HFT)**
- 🔗 Explore more sophisticated copula structures (e.g., **vine copula**, **factor copula**)
- 🧮 Apply to broader financial tasks including **portfolio optimization**, **hedging**, and **risk control**

## 📚 Dependencies

- `numpy`, `pandas`, `matplotlib`
- `torch` for deep RL
- `scipy` for optimization
- `gym`-style custom environments

## 🧑‍💻 Author

**Kevin Long (kevinlmf)**  
Graduate Student @ University of Michigan  
[LinkedIn](https://www.linkedin.com/in/mengfanlong)

---

## 📄 License

MIT License © 2025 Mengfan Long
