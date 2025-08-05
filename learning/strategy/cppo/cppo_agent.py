import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from learning.strategy.cppo.networks import Actor, Critic

class CPPOAgent:
    def __init__(self, state_dim, action_dim, device="cpu", lr=3e-4, gamma=0.99,
                 clip_epsilon=0.2, cvar_alpha=0.05, cvar_weight=1.0, lagrange_lr=0.01):
        self.device = torch.device(device)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.cvar_alpha = cvar_alpha
        self.cvar_weight = cvar_weight
        self.lagrange_lr = lagrange_lr

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        self.lagrange_multiplier = torch.tensor(1.0, requires_grad=False, device=self.device)
        self.clear_buffer()

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            weights = self.actor(state_tensor).squeeze(0)
        return weights.cpu().numpy()

    def store_transition(self, transition):
        state, action, reward, done = transition
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def train_step(self):
        states = torch.tensor(np.array(self.states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(self.actions), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(self.device)

        # Compute returns
        returns, G = [], 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - d)
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        values = self.critic(states).detach()
        advantages = returns - values

        # CVaR estimation
        sorted_returns, _ = torch.sort(returns)
        var_index = int(self.cvar_alpha * len(sorted_returns))
        cvar_value = sorted_returns[:var_index].mean()

        for _ in range(4):  # PPO iterations
            logits = self.actor(states)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            log_probs = dist.log_prob(torch.argmax(actions, dim=1))
            old_log_probs = log_probs.detach()

            ratios = torch.exp(log_probs - old_log_probs)
            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()

            critic_loss = nn.MSELoss()(self.critic(states), returns)
            cvar_penalty = self.lagrange_multiplier * self.cvar_weight * cvar_value
            total_loss = actor_loss + 0.5 * critic_loss + cvar_penalty

            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            total_loss.backward()
            self.actor_opt.step()
            self.critic_opt.step()

        self.lagrange_multiplier += self.lagrange_lr * (cvar_value - 0.0)
        self.lagrange_multiplier = torch.clamp(self.lagrange_multiplier, 0.0)
        self.clear_buffer()

    def clear_buffer(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

