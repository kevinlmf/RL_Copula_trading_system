import os
import numpy as np
import torch
import pandas as pd

from learning.env.copula_trading_env import CopulaTradingEnv
from learning.strategy.sac.sac_agent import SACAgent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📦 Using device: {DEVICE}")

env = CopulaTradingEnv(
    data_path="data/raw_data/real_asset_log_returns_extreme.csv",
    window_size=30,
    initial_cash=1e6,
    copula_type="gaussian"
)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = SACAgent(state_dim, action_dim, device=DEVICE)

EPISODES = 30
MAX_STEPS = 200
returns = []

for episode in range(EPISODES):
    state = env.reset()
    for step in range(MAX_STEPS):
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        agent.store_transition(state, action, reward, next_state, done)
        agent.train_step()

        # ✅ 使用 info 中 return 字段，或者 reward/portfolio normalization
        normalized_return = info.get("return", reward / env.initial_cash)  # fallback
        returns.append(normalized_return)

        state = next_state
        if done:
            break

    print(f"📉 Episode {episode + 1}/{EPISODES} | Total Reward: {round(sum(returns[-MAX_STEPS:]), 2)}")

os.makedirs("results", exist_ok=True)
pd.DataFrame({"Returns": returns}).to_csv("results/sac_returns.csv", index=False)
print("✅ Saved sac_returns.csv")



