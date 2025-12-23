import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from typing import Dict, Optional
import matplotlib.cm as cm


def setup_colors(n_colors: int):
    if n_colors <= 8:
        colors = [
            (1.0, 0.0, 0.0),    
            (0.0, 0.0, 1.0),    
            (1.0, 1.0, 0.0),    
            (0.0, 1.0, 1.0),    
            (1.0, 0.5, 0.0),    
            (0.8, 0.0, 0.8),    
            (0.5, 0.5, 0.0),    
            (0.0, 0.5, 0.5),   
        ]
        cell_cmap = ListedColormap(colors[:n_colors])
    else:
        cell_cmap = cm.get_cmap('hsv')
    

    color_dict = {
        'agent': (0.6, 0.2, 0.8),      
        'goal_marker': (0.2, 0.8, 0.2), 
        'obstacle': (0.3, 0.3, 0.3),    
        'visited': (0.9, 0.9, 0.7),     
        'start': (0.2, 0.5, 0.8),       
        'trajectory': (1.0, 0.5, 0.0),  
        'text': (0.0, 0.0, 0.0),        
    }
    
    return cell_cmap, color_dict


def render(env):
    if env.render_mode == 'ansi':
        return render_text(env)
    elif env.render_mode == 'rgb_array':
        return render_rgb_array(env)
    elif env.render_mode == 'human':
        render_human(env)
    else:
        return render_text(env)


def render_text(env) -> str:
    grid = ""
    for i in range(env.height):
        row = ""
        for j in range(env.width):
            if (i, j) == env.agent_pos:
                row += " A "
            elif (i, j) == env.pos_goal:
                row += " G "
            elif env.obstacle_mask[i, j]:
                row += "███"
            else:
                row += f"{env.colors[i, j]:2d} "
        grid += row + "\n"
    
    info = f"Environment: {env.env_id}\n"
    info += f"Step: {env.steps}/{env.max_steps}, Total Reward: {env.episode_reward:.2f}\n"
    info += f"Agent: {env.agent_pos}, Goal: {env.pos_goal}\n"
    info += f"Visited cells: {len(env.visited_cells)}, Revisits: {env.revisit_count}\n"
    
    return grid + "\n" + info


def render_rgb_array(env) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(max(6, env.width), max(4, env.height)))
    render_to_axis(env, ax, show_trajectory=False)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(height, width, 3)
    
    plt.close(fig)
    return image

_figures = {}


def render_human(env):
    env_id = id(env)
    
    if env_id not in _figures:
        _figures[env_id] = {'fig': None, 'ax': None}
    
    if _figures[env_id]['fig'] is None:
        _figures[env_id]['fig'], _figures[env_id]['ax'] = plt.subplots(
            figsize=(max(8, env.width), max(6, env.height)))
        plt.ion()  
        plt.show()
        plt.close()
    
    ax = _figures[env_id]['ax']
    ax.clear()
    render_to_axis(env, ax, show_trajectory=False)
    

def render_to_axis(env, ax, show_trajectory=False):
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(-0.5, env.height - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis() 
    ax.set_xticks(np.arange(env.width))
    ax.set_yticks(np.arange(env.height))
    ax.grid(True, color='gray', linewidth=0.5, linestyle='-')
    ax.set_axisbelow(True)
    

    for i in range(env.height):
        for j in range(env.width):
            pos = (i, j)
            cell_color_idx = env.colors[i, j]
            
            if env.obstacle_mask[i, j]:
                cell_color = env.color_dict['obstacle']
            elif pos == env.agent_pos or pos == env.pos_goal:
                if env.n_colors > 1:
                    norm_idx = cell_color_idx / (env.n_colors - 1)
                else:
                    norm_idx = 0.5
                
                try:
                    cell_color = env.cell_cmap(norm_idx)
                except:
                    cell_color = (0.8, 0.8, 0.8, 1.0)  
            else:
                if env.n_colors > 1:
                    norm_idx = cell_color_idx / (env.n_colors - 1)
                else:
                    norm_idx = 0.5
                
                try:
                    cell_color = env.cell_cmap(norm_idx)
                except:
                    cell_color = (0.8, 0.8, 0.8, 1.0)  
            
            rect = patches.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                linewidth=1,
                edgecolor='black',
                facecolor=cell_color,
                alpha=0.8
            )
            ax.add_patch(rect)
            
            if not env.obstacle_mask[i, j]:
                brightness = 0.299 * cell_color[0] + 0.587 * cell_color[1] + 0.114 * cell_color[2]
                text_color = 'white' if brightness < 0.5 else 'black'
                
                fontsize = 8 if env.width <= 10 else 6
                ax.text(j, i, str(cell_color_idx), 
                       ha='center', va='center', 
                       fontsize=fontsize,
                       fontweight='bold',
                       color=text_color)
    
    if env.agent_pos is not None:
        agent_y, agent_x = env.agent_pos
        
        triangle_points = [
            (agent_x, agent_y - 0.3),      
            (agent_x - 0.3, agent_y + 0.2),  
            (agent_x + 0.3, agent_y + 0.2)   
        ]
        
        triangle = patches.Polygon(
            triangle_points,
            closed=True,
            color=env.color_dict['agent'],
            edgecolor='black',
            linewidth=2,
            alpha=1.0
        )
        ax.add_patch(triangle)
        
        ax.text(agent_x, agent_y, "A", 
               ha='center', va='center', 
               fontsize=10, fontweight='bold',
               color='white')
        
        agent_color_idx = env.colors[agent_y, agent_x]
        fontsize = 8 if env.width <= 10 else 6
        ax.text(agent_x, agent_y + 0.35, str(agent_color_idx),
               ha='center', va='center',
               fontsize=fontsize,
               fontweight='bold',
               color='black',
               bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))
    
    goal_y, goal_x = env.pos_goal
    
    goal_color_idx = env.colors[goal_y, goal_x]
    fontsize = 8 if env.width <= 10 else 6
    ax.text(goal_x, goal_y, str(goal_color_idx),
           ha='center', va='center',
           fontsize=fontsize,
           fontweight='bold',
           color='white',
           bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.8))
    
    goal_circle = patches.Circle(
        (goal_x, goal_y), 0.25,
        color=env.color_dict['goal_marker'],
        edgecolor='black',
        linewidth=2,
        alpha=1.0
    )
    ax.add_patch(goal_circle)

    ax.text(goal_x, goal_y, "G", 
           ha='center', va='center', 
           fontsize=10, fontweight='bold',
           color='white')
    
    info_text = f"Agent: {env.agent_pos}\n"
    info_text += f"Goal: {env.pos_goal}\n"
    info_text += f"Obstacles: {env.obstacle_mask.sum()}\n"
    info_text += f"Colors: {env.n_colors}\n"
    
    ax.text(1.05, 0.5, info_text, transform=ax.transAxes, 
           fontsize=9, verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)


def create_visualization_png(env, filename: str = "gridworld.png", 
                             show_trajectory: bool = False) -> str:

    fig, ax = plt.subplots(figsize=(max(8, env.width), max(6, env.height)))
    render_to_axis(env, ax, show_trajectory=show_trajectory)
    penalty_text = " (with penalties)" if env.use_penalties else ""
    title = f'GridWorld Environment: {env.env_id}{penalty_text}'
    ax.set_title(title, fontsize=14, pad=15)
    
    legend_elements = [
        patches.Patch(facecolor=env.color_dict['agent'], 
                     label='Agent (purple triangle)'),
        patches.Patch(facecolor=env.color_dict['goal_marker'], 
                     label='Goal (green circle)'),
        patches.Patch(facecolor=env.color_dict['obstacle'], 
                     label='Obstacle'),
    ]
    
    if env.n_colors > 0:
        sample_idx = min(1, env.n_colors - 1)
        if env.n_colors > 1:
            sample_color = env.cell_cmap(sample_idx / (env.n_colors - 1))
        else:
            sample_color = env.cell_cmap(0.5)
        
        legend_elements.append(
            patches.Patch(facecolor=sample_color, 
                         label=f'Floor colors (0-{env.n_colors-1})')
        )
    
    ax.legend(handles=legend_elements, loc='upper left', 
             bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Visualization saved to: {filename}")
    return filename


def close_figure(env):
    env_id = id(env)
    
    if env_id in _figures and _figures[env_id]['fig'] is not None:
        plt.close(_figures[env_id]['fig'])
        _figures[env_id]['fig'] = None
        _figures[env_id]['ax'] = None