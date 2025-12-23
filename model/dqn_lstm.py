import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import time

Transition = namedtuple("Transition", ("state_seq", "action", "next_state_seq", "reward", "done"))


class ReplayMemory:
    
    def __init__(self, capacity, sequence_length=10):
        self.memory = deque([], maxlen=capacity)
        self.sequence_length = sequence_length
        self.current_sequence = deque(maxlen=sequence_length)

    def push(self, state, action, next_state, reward, done):
        self.current_sequence.append((state, action, next_state, reward, done))
        
        if len(self.current_sequence) == self.sequence_length:
            states = [t[0] for t in self.current_sequence]
            actions = [t[1] for t in self.current_sequence]
            next_states = [t[2] for t in self.current_sequence]
            rewards = [t[3] for t in self.current_sequence]
            dones = [t[4] for t in self.current_sequence]
            
            self.memory.append((
                torch.stack(states),
                torch.stack(actions),
                torch.stack(next_states),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.bool)
            ))
            
            self.current_sequence = deque(list(self.current_sequence)[-self.sequence_length//2:], 
                                         maxlen=self.sequence_length)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN_LSTM(nn.Module):
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(DQN_LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
  
        self.fc_in = nn.Linear(input_size, hidden_size)
        
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
  
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.shape

        x = F.relu(self.fc_in(x))
        lstm_out, hidden_out = self.lstm(x, hidden)
        last_out = lstm_out[:, -1, :]
        q_values = self.fc_out(last_out)
        
        return q_values, hidden_out
    
    def init_hidden(self, batch_size=1, device='cpu'):
        
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        )


class LSTMDQNAgent:
    
    def __init__(
        self,
        env,
        lstm_hidden_size=64,
        lstm_num_layers=1,
        learning_rate=1e-4,  
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,  
        epsilon_decay=20000,  
        memory_capacity=20000,
        batch_size=32,
        tau=0.001, 
        sequence_length=10,  
        device=None
    ):
        self.env = env
        self.n_actions = env.action_space.n
        self.n_observations = env.observation_space.shape[0]
        self.sequence_length = sequence_length
        
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.policy_net = DQN_LSTM(
            input_size=self.n_observations,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            output_size=self.n_actions
        ).to(self.device)
        
        self.target_net = DQN_LSTM(
            input_size=self.n_observations,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            output_size=self.n_actions
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.target_net.eval()
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)

        self.memory = ReplayMemory(memory_capacity, sequence_length)
        
        self.steps_done = 0
        self.current_episode_states = []
        self.current_episode_actions = []
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []
        self.epsilon_values = []
    
    def reset_hidden_state(self):
        self.current_episode_states = []
        self.current_episode_actions = []
    
    def select_action(self, state, training=True):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  math.exp(-1.0 * self.steps_done / self.epsilon_decay)
        
        self.current_episode_states.append(state)
        
        if training and random.random() < epsilon:
            action = self.env.action_space.sample()
            self.current_episode_actions.append(action)
            self.steps_done += 1
            return torch.tensor([[action]], device=self.device, dtype=torch.long)
        
        with torch.no_grad():
            if len(self.current_episode_states) < self.sequence_length:
                seq_states = torch.zeros(self.sequence_length, self.n_observations, device=self.device)
                start_idx = self.sequence_length - len(self.current_episode_states)
                for i, s in enumerate(self.current_episode_states):
                    seq_states[start_idx + i] = torch.tensor(s, dtype=torch.float32, device=self.device)
            else:
                seq_states = torch.stack([
                    torch.tensor(s, dtype=torch.float32, device=self.device)
                    for s in self.current_episode_states[-self.sequence_length:]
                ])
            
            seq_states = seq_states.unsqueeze(0)  

            q_values, _ = self.policy_net(seq_states)
            action = q_values.max(1)[1].item()
            
            self.current_episode_actions.append(action)
            self.steps_done += 1
            
            return torch.tensor([[action]], device=self.device, dtype=torch.long)
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        
        transitions = self.memory.sample(self.batch_size)
        state_seqs = torch.stack([t[0] for t in transitions]).to(self.device)
        action_seqs = torch.stack([t[1] for t in transitions]).to(self.device)
        next_state_seqs = torch.stack([t[2] for t in transitions]).to(self.device)
        reward_seqs = torch.stack([t[3] for t in transitions]).to(self.device)
        done_seqs = torch.stack([t[4] for t in transitions]).to(self.device)

        actions = action_seqs[:, -1, 0].long() 
        rewards = reward_seqs[:, -1]  
        dones = done_seqs[:, -1] 
        
        q_values, _ = self.policy_net(state_seqs)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
   
        with torch.no_grad():
            next_q_values, _ = self.target_net(next_state_seqs)
            next_state_values = next_q_values.max(1)[0]
        
        expected_state_action_values = rewards + (1 - dones.float()) * self.gamma * next_state_values
        
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def update_target_network(self):
        with torch.no_grad():
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
    
    def train_episode(self, episode_num, max_steps=1000):
        state, info = self.env.reset()
        
        self.reset_hidden_state()
        
        episode_reward = 0
        episode_length = 0
        episode_loss = 0
        loss_count = 0
        
        for step in range(max_steps):
            action_tensor = self.select_action(state)
            action = action_tensor.item()
            
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            
            self.memory.push(
                state_tensor,
                torch.tensor([action], dtype=torch.long, device=self.device),
                next_state_tensor,
                torch.tensor([reward], dtype=torch.float32, device=self.device),
                torch.tensor([done], dtype=torch.bool, device=self.device)
            )
            
            state = next_state
            
            if len(self.memory) >= self.batch_size and step % 4 == 0: 
                loss = self.optimize_model()
                if loss > 0:
                    episode_loss += loss
                    loss_count += 1
                
                if step % 100 == 0: 
                    self.update_target_network()
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        avg_loss = episode_loss / max(loss_count, 1)
        
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  math.exp(-1.0 * self.steps_done / self.epsilon_decay)

        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_losses.append(avg_loss)
        self.epsilon_values.append(epsilon)
        
        return episode_reward, episode_length, avg_loss
    
    def evaluate(self, n_episodes=10, render=False, max_steps=500):
        total_rewards = []
        successes = []
        
        for episode in range(n_episodes):
            state, info = self.env.reset()
            
            self.reset_hidden_state()
            
            episode_reward = 0
            done = False
            
            if render and episode == 0:
                self.env.render_mode = 'human'
                self.env.render()
                time.sleep(0.5)
            
            for step in range(max_steps):
                with torch.no_grad():
                    if len(self.current_episode_states) < self.sequence_length:
                        seq_states = torch.zeros(self.sequence_length, self.n_observations, device=self.device)
                        start_idx = self.sequence_length - len(self.current_episode_states)
                        for i, s in enumerate(self.current_episode_states):
                            seq_states[start_idx + i] = torch.tensor(s, dtype=torch.float32, device=self.device)
                    else:
                        seq_states = torch.stack([
                            torch.tensor(s, dtype=torch.float32, device=self.device)
                            for s in self.current_episode_states[-self.sequence_length:]
                        ])
                    
                    seq_states = seq_states.unsqueeze(0)
                    q_values, _ = self.policy_net(seq_states)
                    action = q_values.max(1)[1].item()
                
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                self.current_episode_states.append(next_state)
                
                episode_reward += reward
                state = next_state
                
                if render and episode == 0:
                    self.env.render()
                    time.sleep(0.05)
                
                if done:
                    successes.append(terminated)
                    break
            
            total_rewards.append(episode_reward)
        
        if render:
            self.env.render_mode = None
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'success_rate': np.mean(successes) if successes else 0,
            'episode_rewards': total_rewards
        }
    
    def plot_training_history(self, window=50, save_path="lstm_training_history.png"):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        episodes = range(len(self.episode_rewards))
        ax1 = axes[0, 0]
        ax1.plot(episodes, self.episode_rewards, alpha=0.3, label='Raw', color='blue')
        
        if len(self.episode_rewards) >= window:
            smoothed_rewards = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(self.episode_rewards)), smoothed_rewards, 
                    linewidth=2, label=f'Smoothed (window={window})', color='red')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        ax2.plot(episodes, self.episode_lengths, alpha=0.3, color='green')
        if len(self.episode_lengths) >= window:
            smoothed_lengths = np.convolve(self.episode_lengths, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(self.episode_lengths)), smoothed_lengths, 
                    linewidth=2, color='darkgreen')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Length')
        ax2.set_title('Episode Lengths')
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        if self.episode_losses:
            valid_losses = [loss for loss in self.episode_losses if loss > 0]
            valid_episodes = [i for i, loss in enumerate(self.episode_losses) if loss > 0]
            if valid_losses:
                ax3.plot(valid_episodes, valid_losses, alpha=0.6, color='orange')
                ax3.set_xlabel('Episode')
                ax3.set_ylabel('Loss')
                ax3.set_title('Training Loss')
                ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        ax4.plot(episodes, self.epsilon_values, color='purple')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Epsilon')
        ax4.set_title('Exploration Rate (ε)')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'LSTM-DQN Training History - {self.env.env_id}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        
        return fig
    
    def save(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'steps_done': self.steps_done,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_losses': self.episode_losses,
            'epsilon_values': self.epsilon_values,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.episode_losses = checkpoint.get('episode_losses', [])
        self.epsilon_values = checkpoint.get('epsilon_values', [])
        print(f"Model loaded from {path}")