import gym
import numpy as np
import pandas as pd

from learning.copula.gaussian_copula import GaussianCopula
from learning.copula.t_copula import TCopula  # 如果有的话

class CopulaTradingEnv(gym.Env):
    def __init__(self, 
                 data_path="data/raw_data/real_asset_log_returns_extreme.csv",
                 window_size=30,
                 initial_cash=1e6,
                 copula_type="gaussian"):
        super().__init__()
        
        # === Load data ===
        self.df = pd.read_csv(data_path, index_col=0)
        self.data = self.df.values
        self.asset_dim = self.data.shape[1]
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.copula_type = copula_type

        # === Fit Copula ===
        if self.copula_type == "gaussian":
            self.copula = GaussianCopula()
        elif self.copula_type == "t":
            self.copula = TCopula()
        else:
            raise ValueError(f"Unsupported copula type: {self.copula_type}")

        print(f"🔄 Fitting {copula_type.title()} Copula...")
        self.copula.fit(self.data)

        # === Define observation & action space ===
        obs_dim = self.window_size * self.asset_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.asset_dim,), dtype=np.float32
        )

        # === Internal states ===
        self.current_step = None
        self.cash = None
        self.portfolio_value = None
        self.done = False

        print(f"✅ CopulaTradingEnv initialized | obs_dim={obs_dim}, asset_dim={self.asset_dim}")

    def reset(self):
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.portfolio_value = self.cash
        self.done = False
        return self._get_observation()

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset() before using step().")

        # Normalize weights
        weights = np.clip(action, 0, 1).astype(np.float64)
        weights /= np.sum(weights) + 1e-8

        # Asset return
        returns = self.data[self.current_step]
        reward = self.portfolio_value * np.dot(weights, returns)
        self.portfolio_value += reward

        # Step forward
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        obs = self._get_observation()
        info = {"portfolio_value": self.portfolio_value}
        return obs, reward, self.done, info

    def _get_observation(self):
        start = self.current_step - self.window_size
        window = self.data[start:self.current_step].flatten()
        return window.astype(np.float32)
