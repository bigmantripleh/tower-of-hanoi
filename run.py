import numpy as np
import time
from tower_of_hanoi import raw_env

# Create the environment
env = raw_env(
    num_disks=3,
    render_mode="human",      # use "human" for window
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
