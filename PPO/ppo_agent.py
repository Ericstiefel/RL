from actor_critic import Actor, Critic
import gymnasium as gym
import torch
import numpy as np

class PPOagent:
    def __init__(
            self,
            env: gym.Env,
            actor: Actor,
            critic: Critic,
            epsillon: float = 0.2,
            lr: float = 1e-5,
            gamma: float = 0.99
    ):
        self.env = env
        self.Actor = actor
        self.Critic = critic
        self.actor_optimizer = torch.optim.SGD(self.Actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.SGD(self.Critic.parameters(), lr=lr)
        self.epsillon = epsillon
        self.lr = lr
        self.gamma = gamma

    
    def computeDiscountedRewards(self, rewards):
        discounted = []
        Q = 0
        for reward in reversed(rewards):
            Q = reward + self.gamma * Q
            discounted.insert(0, Q)
        return discounted
    
    @torch.no_grad
    def computeAdvantages(self, states, rewards):
        states = torch.tensor(np.array(list(states)), dtype=torch.float32)
        rewards = torch.tensor(np.array(self.computeDiscountedRewards(rewards)), dtype=torch.float32)
        
        normalizedRewards = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        oldValues = self.Critic.forward(states)

        advantages = normalizedRewards - oldValues
        return advantages
        

    def updateActor(self, trajectories):
        for trajectory in trajectories:
        
            states, actions, rewards, old_log_probs = trajectory

            states = torch.tensor(np.array(states), dtype=torch.float32)
            actions = torch.tensor(np.array(actions), dtype=torch.float32)
            with torch.no_grad():
                old_log_probs = torch.tensor(np.array(old_log_probs), dtype=torch.float32)

            _, new_log_probs = self.Actor.sampleAction(states)

            ratios = torch.exp(new_log_probs - old_log_probs)

            advantages = self.computeAdvantages(states, rewards)
            clipped_ratios = torch.clamp(ratios, 1 - self.epsillon, 1 + self.epsillon)
            loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()


    def updateCritic(self, trajectories):
        for states, actions, rewards, log_probs in trajectories:
            returns = torch.tensor(np.array(self.computeDiscountedRewards(rewards)), dtype=torch.float32)
            states = torch.tensor(np.array(states), dtype=torch.float32)

            values = self.Critic(states).squeeze(-1)
            critic_loss = ((returns - values) ** 2).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()


    def nEpisodes(self, n):
        trajectories = []
        for _ in range(n):
            states, actions, rewards, log_probs = [], [], [], []
            state, _ = self.env.reset()
            while True:
                action, log_prob = self.Actor.sampleAction(torch.tensor(np.array(state), dtype=torch.float32))
                next_state, reward, is_done, is_trunc, _ = self.env.step(action)
                states.append(state)
                actions.append(action) 
                rewards.append(reward)
                log_probs.append(log_prob)
                
                if is_done or is_trunc:
                    break
                state = next_state
            trajectories.append((states, actions, rewards, log_probs))
        self.updateActor(trajectories)
        self.updateCritic(trajectories)
        
        return np.mean([rew for s, a, rew, lp in trajectories]) #mean reward of each batch for smoothing