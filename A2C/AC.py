import torch

class ActorCritic(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(ActorCritic, self).__init__()

        self.sharedNN = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 32),
            torch.nn.ReLU()
        )

        self.actor = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_dim),
            torch.nn.Softmax(dim=1)
        )

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        shared_params = self.sharedNN(x)
        actor_probs = self.actor(shared_params)
        state_val = self.critic(shared_params)
        return actor_probs, state_val
        