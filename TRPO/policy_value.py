import torch

class Policy(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()

        self.StochasticPolicyModel = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_dim),
        )

        self.log_std = torch.nn.Parameter(torch.zeros(action_dim))


    def forward(self, state):
        means = self.StochasticPolicyModel(state)
        log_std = self.log_std.expand_as(means)
        stds = torch.exp(log_std)
        return means, stds
    
    def sampleAction(self, state):
        means, stds = self.forward(state)
        distribution = torch.distributions.Normal(means, stds)
        chosen_action = distribution.sample()
        log_prob = distribution.log_prob(chosen_action).sum(dim=-1)
        return chosen_action, log_prob
    
    @torch.no_grad
    def set_parameters(self, new_params):
        """Set the parameters of the model to new_params."""
        for param, new_param in zip(self.parameters(), new_params):
            param.copy_(new_param)


class Value(torch.nn.Module):
    def __init__(self, state_dim):
        super(Value, self).__init__()

        self.ValueModel = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
    def forward(self, state):
        return self.ValueModel(state)
    
    @torch.no_grad
    def set_parameters(self, new_params):
        """Set the parameters of the model to new_params."""
        for param, new_param in zip(self.parameters(), new_params):
            param.copy_(new_param)

