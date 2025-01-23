from net import Net
import gymnasium as gym
import torch
import numpy as np

class ESagent:
    def __init__(self, percentages: Net, env: gym.Env, lr: float = 1e-2, noise_std: float = 0.001):
        self.env = env
        self.percentages = percentages
        self.lr = lr
        self.noise_std = noise_std
    
    def nEpisodes(self, n=10):
        rewards = []
        for _ in range(n):
            obs, _ = self.env.reset()
            totalReward = 0.0
            while True:
                obs_i = torch.FloatTensor(np.expand_dims(obs, 0))
                action_probs = self.percentages.forward(obs_i)
                bestAction = action_probs.max(dim=1)[1]
                obs, reward, is_done, is_trunc, _ = self.env.step(bestAction.data.numpy()[0])
                totalReward += reward
                if is_done or is_trunc:
                    break
            rewards.append(totalReward)
        return rewards
    
    def evalWithNoise(self):
        noise = [torch.tensor(np.random.normal(size=p.data.size()), dtype=torch.float32) for p in self.percentages.parameters()]
        old_params = self.percentages.state_dict()

        for param, ei in zip(self.percentages.parameters(), noise):
            param.data += self.noise_std * ei
        reward = self.nEpisodes()
        self.percentages.load_state_dict(old_params)
        return reward, noise
    
    def trainStep(self):
        rewards, noise = self.evalWithNoise()
        normalizedRewards = rewards - np.mean(rewards)

        weightedNoise = [torch.zeros_like(ei) for ei in noise]

        for ei, reward in zip(noise, normalizedRewards):
            weightedNoise += [reward * ei]

        for p, p_update in zip(self.percentages.parameters(), weightedNoise):
            update = p_update / len(rewards) * self.noise_std
            p.data += self.lr * update

        return rewards

