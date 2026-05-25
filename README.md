# RL Pong Simulation

Two REINFORCE agents learn to play Pong against each other.

## Files
| File | Purpose |
|------|---------|
| `config.py` | Shared constants (screen size, colors, speeds) |
| `rl_agent.py` | PolicyNetwork (MLP) + RLAgent (REINFORCE) |
| `loading_screen.py` | Pygame slider UI to set all parameters |
| `pong_game.py` | Main game loop — run this |

## Install dependencies
```
pip install pygame torch numpy opencv-python
```

## Run
```
python pong_game.py
```

## Loading Screen Parameters
| Param | What it controls |
|-------|-----------------|
| Number of Games | How many episodes to simulate |
| Simulation FPS | Speed of the display (10 = slow-mo, 240 = fast) |
| P1/P2 Learning Rate | RMSprop LR for each agent's policy |
| P1/P2 Gamma | Discount factor for returns |
| P1/P2 Hidden Dim | Width of the hidden layer in the MLP |

## How the RL works
- Each frame is captured from pygame, converted to grayscale, resized to 60×80, and flattened → 4800-dim input vector.
- A 2-layer MLP outputs P(UP) → Bernoulli sample → action.
- Reward: +1 when the agent scores, −1 when it concedes.
- After each game (first to 10 points) REINFORCE updates both policies independently.

## Tips
- Set FPS high (120–240) to train faster with less visual overhead.
- Agents start random; expect ~20–30 games before any coherent tracking appears.
- Frame-differencing (motion signal) is computed but currently the raw gray frame is fed — swap `state_l = agent_l.preprocess(diff)` in `pong_game.py` to use motion.