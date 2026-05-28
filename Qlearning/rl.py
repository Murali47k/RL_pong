"""
rl.py  —  Tabular Q-Learning agents for Pong.

Standard tabular Q-learning with ε-greedy exploration.

    Q(s,a) ← Q(s,a) + α [ r + γ · max_a' Q(s',a') − Q(s,a) ]

α (learning rate) and ε (exploration) both decay over time so the
agent gradually shifts from exploring to exploiting what it learned.
"""

import random
import math
import collections
import numpy as np
import pickle
import os

from config import GAME_H, GAME_W

# Hyper-parameters

N_REL       = 16       # number of buckets for ball-Y relative to paddle-Y
                       # covers range [-GAME_H, +GAME_H]

N_ACTIONS   = 3        # 0=up, 1=down, 2=stay

ALPHA_START = 0.7      # initial learning rate
ALPHA_MIN   = 0.05     # floor for learning rate
ALPHA_DECAY = 4_000    # steps until α ≈ ALPHA_MIN  (exponential)

EPS_START   = 0.8      # initial exploration probability
EPS_MIN     = 0.03     # minimum exploration (always a bit random)
EPS_DECAY   = 6_000   # steps until ε ≈ EPS_MIN

GAMMA       = 0.92     # discount factor

# How often (in simulation steps) each agent is allowed to act & learn.
# At 60 FPS, ACT_EVERY=10 → ~6 decisions/second.
ACT_EVERY   = 10


# State encoding

def _discretize_rel(ball_cy, paddle_cy):
    """
    Map (ball_center_y - paddle_center_y) into one of N_REL buckets.
    Returns an integer in [0, N_REL).
    """
    delta = ball_cy - paddle_cy              # signed, pixels
    # clamp to [-GAME_H, GAME_H] then normalise to [0, 1]
    normalised = (delta + GAME_H) / (2 * GAME_H)
    bucket = int(normalised * N_REL)
    return max(0, min(N_REL - 1, bucket))


def encode_state(ball_cy, paddle_cy, ball_vx, is_left_paddle):
    """
    Return an integer state index.

    ball_dir_toward = 1 if the ball is heading toward THIS paddle, else 0.
    For the left paddle  : vx < 0  means ball coming toward it.
    For the right paddle : vx > 0  means ball coming toward it.
    """
    rel     = _discretize_rel(ball_cy, paddle_cy)
    toward  = int((ball_vx < 0) if is_left_paddle else (ball_vx > 0))
    return rel * 2 + toward          # unique integer for every (rel, toward) pair


N_STATES = N_REL * 2                 # total number of discrete states


# Q-Learning Agent

class QLAgent:
    """
    Tabular Q-learning agent for one Pong paddle.

    Parameters
    ----------
    name          : string label shown in the sidebar
    is_left       : True for the left (P1) paddle
    """

    def __init__(self, name="agent", is_left=True):
        self.name    = name
        self.is_left = is_left

        # Q-table: shape (N_STATES, N_ACTIONS), initialised optimistically
        self.Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float32)

        self.steps    = 0
        self.episodes = 0

        # Moving window for average TD error (shown in sidebar)
        self._td_errors = collections.deque(maxlen=200)

        # Memory for one-step updates
        self._prev_state  = None
        self._prev_action = None

        # Internal step counter: agent only acts every ACT_EVERY sim-steps
        self._step_counter = 0
        self._cached_action = 2    # default: STAY until first real decision

        self.training = True       # set False for eval / greedy play

    # schedules

    @property
    def epsilon(self):
        return EPS_MIN + (EPS_START - EPS_MIN) * math.exp(-self.steps / EPS_DECAY)

    @property
    def alpha(self):
        return ALPHA_MIN + (ALPHA_START - ALPHA_MIN) * math.exp(-self.steps / ALPHA_DECAY)

    @property
    def avg_td_error(self):
        return float(np.mean(self._td_errors)) if self._td_errors else 0.0

    # public API 

    def new_episode(self, ball_cy, paddle_cy, ball_vx):
        """Call at the start of every game round."""
        self.episodes += 1
        self._prev_state  = encode_state(ball_cy, paddle_cy, ball_vx, self.is_left)
        self._prev_action = None
        self._step_counter = 0
        self._cached_action = 2

    def act(self, ball_cy, paddle_cy, ball_vx):
        """
        Called every simulation step.
        Returns an action (0=up, 1=down, 2=stay) but only re-decides
        every ACT_EVERY steps (~4 Hz at 60 FPS).
        """
        self._step_counter += 1

        if self._step_counter < ACT_EVERY:
            return self._cached_action        # keep last decision

        # --- time to make a new decision ---
        self._step_counter = 0
        state = encode_state(ball_cy, paddle_cy, ball_vx, self.is_left)

        # ε-greedy action selection
        if self.training and random.random() < self.epsilon:
            action = random.randrange(N_ACTIONS)
        else:
            action = int(np.argmax(self.Q[state]))

        self._cached_action  = action
        self._prev_state     = state
        self._prev_action    = action
        self.steps          += 1

        return action

    def remember(self, ball_cy, paddle_cy, ball_vx, reward, done):
        """
        Store the transition and perform a Q-learning update.
        Call this every ACT_EVERY steps (or at end-of-round).
        """
        if not self.training or self._prev_state is None or self._prev_action is None:
            return

        next_state = encode_state(ball_cy, paddle_cy, ball_vx, self.is_left)

        # Bellman update
        current_q = self.Q[self._prev_state, self._prev_action]
        if done:
            target_q = reward
        else:
            target_q = reward + GAMMA * float(np.max(self.Q[next_state]))

        td_error = target_q - current_q
        self.Q[self._prev_state, self._prev_action] += self.alpha * td_error
        self._td_errors.append(abs(td_error))

    # persistence 

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"Q": self.Q, "steps": self.steps}, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.Q     = data["Q"]
        self.steps = data.get("steps", 0)