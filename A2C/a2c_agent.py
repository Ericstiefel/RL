from AC import ActorCritic
import torch
import gymnasium as gym

class A2C_Agent:
    def __init__(self, 
                 Actor_Critic: ActorCritic,
                   gamma: float = 0.99, lr: float = 1e-4, 
                   add_entropy: bool = False, 
                   entropy_multiplier: float = 0.01
                   ):
        
        self.Actor_Critic = Actor_Critic
        self.optimizer = torch.optim.SGD(Actor_Critic.parameters(), lr=lr)
        self.gamma = gamma
        self.add_entropy = add_entropy
        self.entropy_multiplier = entropy_multiplier

    def Entropy(self, dist: torch.tensor):
        return -(dist * torch.log(dist + 1e-10)).sum(dim=1).mean()
    
    def AdvantagesAndReturns(self, rewards, dones, values):
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            G = reward + self.gamma * G * (1 - done)
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = returns - torch.tensor(values, dtype=torch.float32)
        return returns, advantages

    def update(self, states, actions, returns, advantages):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        advantages = advantages.detach()

        action_probs, values = self.Actor_Critic(states)
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())

        policy_loss = -(action_log_probs * advantages).mean()
        value_loss = torch.nn.functional.mse_loss(values.squeeze(), returns)
        loss = policy_loss + value_loss - (self.Entropy(action_probs) * self.add_entropy * self.entropy_multiplier)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def nEpisodes(self, env: gym.Env, n: int) -> float:
        episode_rewards = []
        states, actions, rewards, dones, values = [], [], [], [], []
        for _ in range(1, n+1):
            episode_reward = 0
            state, _ = env.reset()
            while True:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                actor_probs, state_val = self.Actor_Critic.forward(state_tensor)

                chosen_action = torch.multinomial(actor_probs, 1).item()

                next_state, reward, is_done, is_trunc, _ = env.step(chosen_action)
                states.append(state)
                actions.append(chosen_action)
                rewards.append(reward)
                dones.append(is_done or is_trunc)
                values.append(state_val.item())
                episode_reward += reward

                if is_done or is_trunc:
                    break

                state = next_state

            returns, advantages = self.AdvantagesAndReturns(rewards, dones, values)

            self.update(states, actions, returns, advantages)

            episode_rewards.append(episode_reward)
        
        return sum(episode_rewards) / n
