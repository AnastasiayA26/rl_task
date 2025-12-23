import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.my_gridworld import EnhancedGridWorldEnv

os.makedirs("env_examples", exist_ok=True)


colors = np.arange(25).reshape(5, 5)
env1 = EnhancedGridWorldEnv(
    height=5, width=5, colors=colors, n_colors=25,
    pos_goal=(4,4), pos_agent=(0,0), env_id="Env1"
)
env1.reset()
env1.create_visualization_png("env_examples/env1.png")
env1.close()


env2 = EnhancedGridWorldEnv(
    height=5, width=5, n_colors=5,
    pos_goal=(4,4), pos_agent=(0,0), env_id="Env2", seed=42
)
env2.reset()
env2.create_visualization_png("env_examples/env2.png")
env2.close()

np.random.seed(42)
obs_mask = np.random.rand(10, 10) < 0.10
obs_mask[0,0] = obs_mask[9,9] = False
env3 = EnhancedGridWorldEnv(
    height=10, width=10, n_colors=7, obstacle_mask=obs_mask,
    pos_goal=(9,9), pos_agent=(0,0), env_id="Env3", seed=42
)
env3.reset()
env3.create_visualization_png("env_examples/env3.png")
env3.close()

np.random.seed(43)
obs_mask = np.random.rand(10, 10) < 0.10
obs_mask[0,0] = obs_mask[9,9] = False
env4 = EnhancedGridWorldEnv(
    height=10, width=10, n_colors=4, obstacle_mask=obs_mask,
    pos_goal=(9,9), pos_agent=(0,0), env_id="Env4", seed=42
)
env4.reset()
env4.create_visualization_png("env_examples/env4.png")
env4.close()

print("Done! Check the 'results' folder.")