from policy_nn import PolicyNN
import torch
import gymnasium as gym


class Reinforce:

    def __init__(self, state_dim: int, action_dim: int, gamma: float = 0.99, lr: float = 1e-3):
        self.Policy = PolicyNN(state_dim, action_dim)
        self.Optimizer: torch.optim.Optimizer = torch.optim.Adam(self.Policy.parameters(), lr=lr)
        self.gamma = gamma
    
    def updatePolicy(self, log_probs, returns):
        loss = -torch.sum(torch.stack(log_probs) * returns)
        self.Optimizer.zero_grad()
        loss.backward()
        self.Optimizer.step()

    def computeReturns(self, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.FloatTensor(returns)

    def playEpisode(self, env: gym.Env, learn: bool=True):
        log_probabilities = []
        rewards = []
        state, _ = env.reset()

        while True:
            action_choice, log_probs = self.Policy.act(state)
            next_state, reward, is_done, is_trunc, _ = env.step(action_choice)
            rewards.append(reward)
            log_probabilities.append(log_probs)

            if is_done or is_trunc:
                if learn:
                    returns = self.computeReturns(rewards)
                    self.updatePolicy(log_probabilities, returns)
                break

            state = next_state

        return sum(rewards)