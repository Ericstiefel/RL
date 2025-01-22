import torch

class Actor(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Actor, self).__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_dim)
        )

        self.log_std = torch.nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        mean = self.actor(state)
        std = torch.exp(self.log_std)
        return mean, std

    def sampleAction(self, state):
        mean, std = self.forward(state)
        std = torch.maximum(std, torch.tensor(1e-5))
        dist = torch.distributions.Normal(mean, std)
        chosen_action = dist.sample()
        log_prob = dist.log_prob(chosen_action).sum(dim=-1)
        return chosen_action, log_prob
    

class Critic(torch.nn.Module):
    def __init__(self, state_dim: int):
        super(Critic, self).__init__()
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
    def forward(self, state):
        return self.critic(state)
