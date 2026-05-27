"""
rl.py — Lightweight DQN agents for Pong.

Each paddle gets its own DQNAgent that:
  • receives a stack of 6 grayscale 84×84 frames as state
  • outputs Q-values for 2 actions: 0=up, 1=down  (stay removed)
  • trains from a replay buffer using the Bellman equation

No GPU required — the network is small enough for CPU.
"""

import random
import math
import collections
import numpy as np

# ── optional torch import (graceful fallback) ──────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


# ──────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────

FRAME_H    = 84          # resized frame height fed to CNN
FRAME_W    = 84          # resized frame width
STACK_N    = 6           # frames stacked — more context for velocity estimation
N_ACTIONS  = 2           # 0=up, 1=down  (stay removed to prevent idle policies)

BATCH_SIZE     = 64
REPLAY_MAXLEN  = 50_000  # larger buffer → more diverse experience
GAMMA          = 0.995   # higher discount → agent values future rallies more
LR             = 1e-4    # Adam learning rate

EPS_START  = 1.0
EPS_END    = 0.05
EPS_DECAY  = 100_000     # slow decay so agents explore long enough

TARGET_UPDATE_FREQ = 1000  # steps between target-net syncs
TRAIN_FREQ         = 4     # train every N steps
MIN_REPLAY         = 2000  # minimum transitions before training starts


# ──────────────────────────────────────────────────────────────────────────
# CNN Q-Network
# ──────────────────────────────────────────────────────────────────────────

class QNet(nn.Module):
    """
    Small CNN: 6 stacked 84×84 grayscale frames → 2 Q-values (up/down).
    Inspired by the original DQN paper but scaled down for CPU.
    """

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(STACK_N, 16, kernel_size=8, stride=4),  # → 16×20×20
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),        # → 32×9×9
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),        # → 32×7×7
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, N_ACTIONS),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# ──────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ──────────────────────────────────────────────────────────────────────────

Transition = collections.namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:

    def __init__(self, maxlen=REPLAY_MAXLEN):
        self.buf = collections.deque(maxlen=maxlen)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, n):
        return random.sample(self.buf, n)

    def __len__(self):
        return len(self.buf)


# ──────────────────────────────────────────────────────────────────────────
# Frame pre-processing
# ──────────────────────────────────────────────────────────────────────────

def preprocess(surface):
    """
    Convert a pygame Surface (game area only) to a numpy float32 array
    of shape (84, 84), values in [0, 1].
    Requires pygame; called from main.py.
    """
    import pygame

    small = pygame.transform.scale(surface, (FRAME_W, FRAME_H))
    arr   = pygame.surfarray.array3d(small)          # (W, H, 3)
    arr   = arr.transpose(1, 0, 2)                   # (H, W, 3)
    gray  = (
        0.2989 * arr[:, :, 0]
        + 0.5870 * arr[:, :, 1]
        + 0.1140 * arr[:, :, 2]
    ).astype(np.float32) / 255.0                     # (H, W)
    return gray


# ──────────────────────────────────────────────────────────────────────────
# Frame Stack
# ──────────────────────────────────────────────────────────────────────────

class FrameStack:
    """Keeps the last STACK_N frames and returns them as a (STACK_N, H, W) array."""

    def __init__(self):
        self.frames = collections.deque(maxlen=STACK_N)

    def reset(self, frame):
        for _ in range(STACK_N):
            self.frames.append(frame)

    def push(self, frame):
        self.frames.append(frame)

    def state(self):
        return np.stack(self.frames, axis=0)  # (STACK_N, H, W)


# ──────────────────────────────────────────────────────────────────────────
# DQN Agent
# ──────────────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    One DQN agent controlling one paddle.

    If PyTorch is unavailable the agent falls back to random actions so the
    game still runs — you just won't get learning.
    """

    def __init__(self, name="agent"):
        self.name       = name
        self.steps      = 0
        self.episodes   = 0
        self.losses     = collections.deque(maxlen=100)
        self.epsilon    = EPS_START
        self.training   = True          # set False for eval-only mode

        self.frame_stack = FrameStack()

        if not TORCH_OK:
            self.online = self.target = self.opt = self.buf = None
            return

        self.online = QNet()
        self.target = QNet()
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.opt = optim.Adam(self.online.parameters(), lr=LR)
        self.buf = ReplayBuffer()

        self._prev_state  = None
        self._prev_action = None

    # ── helpers ───────────────────────────────────────────────────────────

    def _eps(self):
        """Linearly decayed epsilon."""
        return EPS_END + (EPS_START - EPS_END) * math.exp(
            -self.steps / EPS_DECAY
        )

    def _state_tensor(self, arr):
        """numpy (4,84,84) → torch (1,4,84,84) float32."""
        return torch.from_numpy(arr).unsqueeze(0)

    # ── public API ────────────────────────────────────────────────────────

    def new_episode(self, frame):
        """Call at the start of each game round with the first frame."""
        self.episodes += 1
        self.frame_stack.reset(frame)
        self._prev_state  = self.frame_stack.state()
        self._prev_action = None

    def act(self, frame):
        """
        Given the current frame, push it into the stack and return an action.
        0=up, 1=down
        """
        self.frame_stack.push(frame)
        state = self.frame_stack.state()

        if not TORCH_OK or (self.training and random.random() < self._eps()):
            action = random.randrange(N_ACTIONS)
        else:
            with torch.no_grad():
                q = self.online(self._state_tensor(state))
            action = int(q.argmax(dim=1).item())

        self._prev_state  = state
        self._prev_action = action
        self.steps       += 1
        self.epsilon      = self._eps()

        return action

    def remember(self, frame, reward, done):
        """
        Store transition after act() was called.
        frame   — next observation (raw gray frame)
        reward  — scalar reward for the step
        done    — True if the round ended
        """
        if not TORCH_OK or self._prev_state is None:
            return

        self.frame_stack.push(frame)
        next_state = self.frame_stack.state()

        self.buf.push(
            self._prev_state,
            self._prev_action,
            reward,
            next_state,
            done,
        )

        if (
            self.training
            and len(self.buf) >= MIN_REPLAY
            and self.steps % TRAIN_FREQ == 0
        ):
            self._train_step()

        if self.steps % TARGET_UPDATE_FREQ == 0:
            self.target.load_state_dict(self.online.state_dict())

    def save(self, path):
        if TORCH_OK and self.online:
            torch.save(self.online.state_dict(), path)

    def load(self, path):
        if TORCH_OK and self.online:
            self.online.load_state_dict(
                torch.load(path, map_location="cpu")
            )
            self.target.load_state_dict(self.online.state_dict())

    @property
    def avg_loss(self):
        return sum(self.losses) / len(self.losses) if self.losses else 0.0

    # ── private ───────────────────────────────────────────────────────────

    def _train_step(self):
        batch = self.buf.sample(BATCH_SIZE)

        states      = torch.from_numpy(np.stack([t.state      for t in batch]))
        actions     = torch.tensor([t.action    for t in batch], dtype=torch.long)
        rewards     = torch.tensor([t.reward    for t in batch], dtype=torch.float32)
        next_states = torch.from_numpy(np.stack([t.next_state for t in batch]))
        dones       = torch.tensor([t.done      for t in batch], dtype=torch.float32)

        # current Q-values for chosen actions
        q_current = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # target Q-values (Bellman)
        with torch.no_grad():
            q_next = self.target(next_states).max(dim=1).values
            q_target = rewards + GAMMA * q_next * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q_current, q_target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        self.losses.append(loss.item())