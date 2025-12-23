import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')
import gymnasium as gym
from gymnasium import spaces


class EnhancedGridWorldEnv(gym.Env):  
    metadata = {'render_modes': ['ansi', 'human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(
        self,
        height: int = 5,
        width: int = 5,
        obstacle_mask: Optional[np.ndarray] = None,
        pos_goal: Tuple[int, int] = (4, 4),
        pos_agent: Union[Tuple[int, int], np.ndarray] = (0, 0),
        colors: Optional[np.ndarray] = None,
        n_colors: int = 5,
        see_obstacle: bool = True,
        max_steps: int = 50,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        use_penalties: bool = False,
        goal_reward: float = 1.0,
        step_penalty: float = -0.01,
        revisit_penalty: float = -0.08,
        wall_penalty: float = -0.03,
        env_id: str = "env",
    ):

        super().__init__()

        self.height = height
        self.width = width
        self.n_colors = n_colors
        self.see_obstacle = see_obstacle
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.pos_goal = pos_goal
        self.env_id = env_id

        self.use_penalties = use_penalties
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.revisit_penalty = revisit_penalty
        self.wall_penalty = wall_penalty
        
        self.np_random = np.random.RandomState(seed)

        if obstacle_mask is None:
            self.obstacle_mask = np.zeros((height, width), dtype=bool)
        else:
            self.obstacle_mask = obstacle_mask.astype(bool)

        if (pos_goal[0] < 0 or pos_goal[0] >= height or 
            pos_goal[1] < 0 or pos_goal[1] >= width):
            raise ValueError(f"Goal position {pos_goal} is outside grid bounds")
        if self.obstacle_mask[pos_goal]:
            raise ValueError("Goal cannot be placed on an obstacle")
   
        if isinstance(pos_agent, tuple):
            self.fixed_start = pos_agent
            self.start_positions = None
            
            if (pos_agent[0] < 0 or pos_agent[0] >= height or 
                pos_agent[1] < 0 or pos_agent[1] >= width):
                raise ValueError(f"Start position {pos_agent} is outside grid bounds")
            if self.obstacle_mask[pos_agent]:
                raise ValueError("Start position cannot be on an obstacle")
            if pos_agent == pos_goal:
                raise ValueError("Start position cannot be the same as goal")
        else:
            self.fixed_start = None
            self.start_positions = pos_agent.copy().astype(float)
            self.start_positions[self.obstacle_mask] = 0
            self.start_positions[pos_goal[0], pos_goal[1]] = 0
            total = self.start_positions.sum()
            
            if total > 0:
                self.start_positions /= total
            else:
                self.start_positions = None

        if colors is None:
            self.colors = self.np_random.randint(0, n_colors, size=(height, width))
            self.colors[self.obstacle_mask] = 0  
        else:
            self.colors = colors.copy()

        self.action_space = spaces.Discrete(4) 
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(n_colors + 3,), 
            dtype=np.float32
        )
        
        self.agent_pos = None
        self.steps = 0
        self.episode_reward = 0
        self.visited_cells = set()
        self.revisit_count = 0
        self.trajectory = []
        self.last_attempted_pos = None
        

        from .utils import setup_colors
        self.cell_cmap, self.color_dict = setup_colors(self.n_colors)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):

        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        self.visited_cells.clear()
        self.revisit_count = 0
        self.trajectory = []
        self.last_attempted_pos = None

        if self.fixed_start is not None:
            self.agent_pos = self.fixed_start
        elif self.start_positions is not None:
     
            flat_probs = self.start_positions.flatten()
            if flat_probs.sum() > 0:
                idx = self.np_random.choice(len(flat_probs), p=flat_probs)
                self.agent_pos = (idx // self.width, idx % self.width)
            else:
                self.agent_pos = self._get_random_free_cell()
        else:
            if not self.obstacle_mask[0, 0] and (0, 0) != self.pos_goal:
                self.agent_pos = (0, 0)
            else:
                self.agent_pos = self._get_random_free_cell()
        
        self.visited_cells.add(self.agent_pos)
        self.trajectory.append(self.agent_pos)

        self.steps = 0
        self.episode_reward = 0
        observation = self._get_observation(self.agent_pos)
        
        info = {
            'agent_pos': self.agent_pos,
            'goal_pos': self.pos_goal,
            'distance_to_goal': self._manhattan_distance(self.agent_pos, self.pos_goal),
            'visited_cells': len(self.visited_cells),
        }
        
        if self.render_mode == 'human':
            self.render()
        
        return observation, info
    
    def _get_random_free_cell(self) -> Tuple[int, int]:
        free_cells = []
        for i in range(self.height):
            for j in range(self.width):
                if (not self.obstacle_mask[i, j] and 
                    (i, j) != self.pos_goal):
                    free_cells.append((i, j))
        
        if free_cells:
            return free_cells[self.np_random.randint(len(free_cells))]
        else:
            return (0, 0)
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def step(self, action: int):
        current_row, current_col = self.agent_pos
        
        if action == 0:    # Up
            new_pos = (current_row - 1, current_col)
        elif action == 1:  # Right
            new_pos = (current_row, current_col + 1)
        elif action == 2:  # Down
            new_pos = (current_row + 1, current_col)
        elif action == 3:  # Left
            new_pos = (current_row, current_col - 1)
        else:
            raise ValueError(f"Invalid action: {action}. Must be 0-3.")
        
        self.last_attempted_pos = new_pos
        hit_obstacle = False
        if (new_pos[0] < 0 or new_pos[0] >= self.height or 
            new_pos[1] < 0 or new_pos[1] >= self.width):
            hit_obstacle = True 
        elif self.obstacle_mask[new_pos]:
            hit_obstacle = True 
        
        reward = 0.0
        
        if self.use_penalties:
            reward += self.step_penalty  
            
            if not hit_obstacle:
                old_pos = self.agent_pos
                self.agent_pos = new_pos
                
                if self.agent_pos in self.visited_cells:
                    reward += self.revisit_penalty
                    self.revisit_count += 1
                
                self.visited_cells.add(self.agent_pos)
                self.trajectory.append(self.agent_pos)
            else:
                reward += self.wall_penalty
        else:
            if not hit_obstacle:
                self.agent_pos = new_pos
                self.visited_cells.add(self.agent_pos)
                self.trajectory.append(self.agent_pos)
        
        self.steps += 1
        
        terminated = (self.agent_pos == self.pos_goal)
        truncated = (self.steps >= self.max_steps)
        
        if terminated:
            reward += self.goal_reward
        
        if hit_obstacle and self.see_obstacle:
            observation = self._get_observation(self.last_attempted_pos)
        else:
            observation = self._get_observation(self.agent_pos)
        

        self.episode_reward += reward
        
        info = {
            'agent_pos': self.agent_pos,
            'goal_pos': self.pos_goal,
            'hit_obstacle': hit_obstacle,
            'terminated': terminated,
            'truncated': truncated,
            'distance_to_goal': self._manhattan_distance(self.agent_pos, self.pos_goal),
            'step_reward': reward,
            'total_reward': self.episode_reward,
            'visited_cells': len(self.visited_cells),
            'revisit_count': self.revisit_count,
            'attempted_pos': self.last_attempted_pos if hit_obstacle else None,
            'steps': self.steps,
        }
        
        if self.render_mode == 'human':
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self, pos: Tuple[int, int]) -> np.ndarray:
        obs = np.zeros(self.n_colors + 3, dtype=np.float32)
        
        if pos == self.pos_goal:
            obs[self.n_colors] = 1.0  
        elif (pos[0] < 0 or pos[0] >= self.height or 
              pos[1] < 0 or pos[1] >= self.width):
            obs[self.n_colors + 1] = 1.0  
        elif self.obstacle_mask[pos[0], pos[1]]:
            obs[self.n_colors + 2] = 1.0  
        else:
            color_idx = self.colors[pos[0], pos[1]]
            if 0 <= color_idx < self.n_colors:
                obs[color_idx] = 1.0
            else:
                obs[0] = 1.0
        
        return obs
    
    def render(self):
        from .utils import render
        return render(self)
    
    def create_visualization_png(self, filename: str = "gridworld.png", 
                                 show_trajectory: bool = False) -> str:
        from .utils import create_visualization_png

        return create_visualization_png(self, filename, show_trajectory)
    
    def close(self):
        from .utils import close_figure
        close_figure(self)