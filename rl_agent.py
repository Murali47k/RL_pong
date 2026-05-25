# rl_agent.py — Policy Network + REINFORCE Agent

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

FRAME_H, FRAME_W = 60, 80   # downsample resolution
INPUT_DIM = FRAME_H * FRAME_W


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)   # P(UP)


class RLAgent:
    def __init__(self, lr=1e-3, gamma=0.99, hidden_dim=256, name="Agent"):
        self.name = name
        self.gamma = gamma
        self.policy = PolicyNetwork(INPUT_DIM, hidden_dim)
        self.optimizer = optim.RMSprop(self.policy.parameters(), lr=lr)
        self.log_probs = []
        self.rewards   = []
        self.episode_rewards = []   # total reward per episode


    def preprocess(self, frame_gray):
        """
        frame_gray: H x W uint8 numpy array (grayscale, 0-255)
        Returns flat float32 tensor normalised to [0,1].
        """
        import cv2
        small = cv2.resize(frame_gray, (FRAME_W, FRAME_H),
                           interpolation=cv2.INTER_AREA)
        return torch.from_numpy(small.astype(np.float32).ravel() / 255.0)

    # ------------------------------------------------------------------
    def select_action(self, state_tensor):
        """state_tensor: flat float32 tensor from preprocess()"""
        prob_up = self.policy(state_tensor)
        dist    = torch.distributions.Bernoulli(prob_up)
        action  = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return int(action.item())   # 1 = UP, 0 = DOWN

    # ------------------------------------------------------------------
    def record_reward(self, r):
        self.rewards.append(float(r))

    # ------------------------------------------------------------------
    def finish_episode(self):
        """REINFORCE policy gradient update."""
        R, returns = 0.0, []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = torch.stack([-lp * G for lp, G in zip(self.log_probs, returns)]).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        ep_total = sum(self.rewards)
        self.episode_rewards.append(ep_total)
        self.log_probs = []
        self.rewards   = []
        return ep_total