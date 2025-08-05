import numpy as np

def compute_metrics(portfolio_values, risk_free_rate=0.0):
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe = (mean_return - risk_free_rate) / (std_return + 1e-8)
    max_drawdown = max(1 - portfolio_values[i] / max(portfolio_values[:i+1]) for i in range(1, len(portfolio_values)))

    annual_factor = 252
    return {
        "Final Value": portfolio_values[-1],
        "Annual Return": (1 + mean_return)**annual_factor - 1,
        "Volatility": std_return * np.sqrt(annual_factor),
        "Sharpe": sharpe * np.sqrt(annual_factor),
        "Max Drawdown": max_drawdown,
    }
