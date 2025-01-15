import torch
import numpy as np

class PolicyNN(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNN, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_size, state_size * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(state_size * 2, action_size * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(action_size * 2, action_size),
            torch.nn.ReLU(),
            torch.nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.model(x)
    
    def act(self, state):
        torched_state = torch.from_numpy(state).unsqueeze(0)
        probs = self.forward(torched_state)
        a = torch.distributions.Categorical(probs)
        action = a.sample()
        return action.item(), a.log_prob(action)
