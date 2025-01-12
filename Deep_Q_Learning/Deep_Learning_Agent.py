from collections import deque
import numpy as np
import torch
from Conv import CNN
import random

State = np.ndarray
Action = int
Reward = float

class DQNAgent:
    def __init__(
        self,
        state_shape=(84, 84, 4),
        action_dim=6,
        lr=0.00025,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        replay_buffer_size=10000,
        batch_size=32,
        target_update_freq=1000,
    ):
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        self.q_network = CNN(input_shape=state_shape, output_shape=(action_dim,))
        self.target_network = CNN(input_shape=state_shape, output_shape=(action_dim,))
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.step_count = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.q_network(state)).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_experiences(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),       
            torch.tensor(actions, dtype=torch.int64),    
            torch.tensor(rewards, dtype=torch.float32),     
            torch.tensor(next_states, dtype=torch.float),   
            torch.tensor(dones, dtype=torch.int64)         
        )

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_experiences()
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0]

        target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = torch.nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.step_count += 1

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def play_episode(self, env):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = self.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            self.store_experience(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        self.decay_epsilon()
        return total_reward

