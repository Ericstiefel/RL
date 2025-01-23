import torch

class Net(torch.nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super(Net, self).__init__()

        self.evaluate = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, action_dim),
            torch.nn.Softmax(dim=1)
        )   
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.evaluate(x)
        