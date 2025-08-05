import os
import numpy as np
import torch
import pandas as pd

from learning.env.copula_trading_env import CopulaTradingEnv
from learning.strategy.cppo.cppo_agent import CPPOAgent

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
agent = CPPOAgent(state_dim=state_dim, action_dim=action_dim, device=DEVICE)

EPISODES = 30
MAX_STEPS = 200
returns = []

for episode in range(EPISODES):
    state = env.reset()
    for step in range(MAX_STEPS):
        weights = agent.select_action(state)
        next_state, reward, done, info = env.step(weights)

        agent.store_transition((state, weights, reward, done))

        normalized_return = info.get("return", reward / env.initial_cash)
        returns.append(normalized_return)

        state = next_state
        if done:
            break

    agent.train_step()
    print(f"📈 Episode {episode + 1}/{EPISODES} | Total Reward: {round(sum(returns[-MAX_STEPS:]), 2)}")

os.makedirs("results", exist_ok=True)
pd.DataFrame({"Returns": returns}).to_csv("results/cppo_returns.csv", index=False)
print("✅ Saved cppo_returns.csv")



