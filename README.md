# Tower of Hanoi

A minimal **single-agent PettingZoo AEC environment** implementing the classic **Tower of Hanoi** puzzle, with action masking and optional pygame rendering.

---

## Environment Details

- **Agents:** 1 (`player_0`)
- **Action space:** `Discrete(6)` (directed moves between pegs)
- **Observation space:** `Dict`
  - `observation`: `(3, num_disks)` tower representation (`int8`)
  - `action_mask`: `(6,)` binary mask of legal actions
- **Rendering modes:** `None`, `human`, `rgb_array`

---

## Rewards

- `-1.0` for illegal actions  
- `-0.3` per step **after exceeding the optimal solution length**  
- `+10.0` for solving the puzzle  

Optimal number of steps: 2**num_disks - 1

---

## Usage

```python
import numpy as np
import time
from tower_of_hanoi import raw_env

# Create the environment
env = raw_env(
    num_disks=3,
    render_mode="None",      # use "human" for window
    manual_control=False
)

env.reset()

total_reward = 0.0

for agent in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    total_reward += reward

    if term or trunc:
        action = None
    else:
        # pick a random legal action
        legal = np.where(obs["action_mask"] == 1)[0]
        action = np.random.choice(legal)

    env.step(action)

    time.sleep(0.5)  # slow down

env.close()
print(f"Episode finished. Total reward: {total_reward}")

## AI Co-Creation Notice

This project was co-created by human authors with the assistance of AI tools.
AI support was used for code creation, review, refactoring, and documentation drafting.