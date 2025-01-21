from policy_value import Policy, Value
import gymnasium as gym
import torch


class TRPO:
    def __init__(
            self,
            policyAgent: Policy,
            valueAgent: Value,
            env: gym.Env,
            delta: float = 0.01,
            gamma: float = 0.99,
            lam: float = 0.97
            ):

        self.policyAgent = policyAgent
        self.valueAgent = valueAgent
        self.env = env
        self.delta = delta
        self.gamma = gamma
        self.lam = lam
    
    def computeDiscountedRewards(self, rewards):
        discounted = []
        Q = 0
        for reward in reversed(rewards):
            Q = reward + self.gamma * Q
            discounted.insert(0, Q)
        return discounted

    def computeAdvantages(self, trajectories):
        advantages, returns = [], []
        for state, _, reward in trajectories:
            dis_rewards = torch.tensor(self.computeDiscountedRewards(reward), dtype=torch.float32)
            returns.extend(dis_rewards)
            values = self.valueAgent(torch.tensor(state, dtype=torch.float32)).squeeze(-1)
            adv = dis_rewards - values
            advantages.extend(adv.detach().numpy())
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)
    
    def update(self, trajectories):
        def compute_kl_divergence(states):
            """Computes the KL divergence between old and new policies."""
            new_means, new_stds = self.policyAgent(states)
            
            old_means, old_stds = means.detach(), stds.detach()
            

            kl = torch.log(new_stds / old_stds) + (
                old_stds.pow(2) + (old_means - new_means).pow(2)
            ) / (2.0 * new_stds.pow(2)) - 0.5
            return kl.sum(dim=-1).mean()

        def surrogate_loss(states, actions, advantages, old_log_probs):
            """Surrogate loss for policy improvement."""
            new_means, new_stds = self.policyAgent(states)
            dist = torch.distributions.Normal(new_means, new_stds)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            ratio = torch.exp(new_log_probs - old_log_probs)
            return -(ratio * advantages).mean()

        def conjugate_gradient(Ax, b, n_steps=10, residual_tol=1e-10):
            """Conjugate gradient to solve Ax = b."""
            x = torch.zeros_like(b)
            r = b.clone()
            p = b.clone()
            r_dot_old = torch.dot(r, r)

            for _ in range(n_steps):
                Ap = Ax(p)
                alpha = r_dot_old / torch.dot(p, Ap)
                x = x * (alpha * p)
                r = r - (alpha * Ap)
                r_dot_new = torch.dot(r, r)
                if r_dot_new < residual_tol:
                    break
                p = r + (r_dot_new / r_dot_old) * p
                r_dot_old = r_dot_new
            return x

        def line_search(step_dir, max_kl, old_params, states):
            """Performs line search to satisfy KL constraint."""
            step_size = 1.0
            for _ in range(5):  # Limit the number of backtracking steps
                new_params = old_params + step_size * step_dir
                self.policyAgent.set_parameters(new_params)
                kl_div = compute_kl_divergence(states)
                if kl_div <= max_kl:
                    return True
                step_size *= 0.5  
            return False

        def fisher_vector_product(v):
            kl = compute_kl_divergence(states)
            torch.autograd.set_detect_anomaly(True)

            grads_kl = torch.autograd.grad(kl, self.policyAgent.parameters(), create_graph=True)
            grads_kl = torch.cat([grad.view(-1) for grad in grads_kl])
            fvp = torch.autograd.grad(torch.dot(grads_kl, v), self.policyAgent.parameters(), retain_graph=True)
            return torch.cat([grad.contiguous().view(-1) for grad in fvp]) + 0.1 * v
        
        advantages, returns = self.computeAdvantages(trajectories)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.cat([torch.tensor(t[0], dtype=torch.float32) for t in trajectories])
        actions = torch.cat([torch.tensor(t[1], dtype=torch.float32) for t in trajectories])

        means, stds = self.policyAgent(states)
        dist = torch.distributions.Normal(means, stds)
        old_log_probs = dist.log_prob(actions).sum(dim=-1).detach()

        loss = surrogate_loss(states, actions, advantages, old_log_probs)
        grads = torch.autograd.grad(loss, self.policyAgent.parameters(), create_graph=True)
        grads = torch.cat([grad.view(-1) for grad in grads])


        step_dir = conjugate_gradient(fisher_vector_product, -grads)
        shs = 0.5 * torch.dot(step_dir, fisher_vector_product(step_dir))
        lm = torch.sqrt(shs / self.delta)
        step_size = 1 / lm

        old_params = torch.cat([param.view(-1) for param in self.policyAgent.parameters()])
        if not line_search(step_dir * step_size, self.delta, old_params, states):
            self.policyAgent.set_parameters(old_params)

        # Value function update
        value_loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.valueAgent.parameters(), lr=1e-3)

        for _ in range(5):  # Multiple epochs of value function update
            values = self.valueAgent(states).squeeze(-1)
            loss = value_loss(values, returns)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def nEpisodes(self, n):
        trajectories = [] #size (n, (state, action, reward))
        for _ in range(n):
            states, actions, rewards = [], [], []
            state, _ = self.env.reset()
            while True:
                action, _ = self.policyAgent.sampleAction(torch.tensor(state, dtype=torch.float32))
                action = action.numpy()
                next_state, reward, is_done, is_trunc, _ = self.env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                if is_done or is_trunc:
                    break
                state = next_state
            trajectories.append((states, actions, rewards))
        self.update(trajectories)
        return trajectories