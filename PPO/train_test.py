from actor_critic import Actor, Critic
from ppo_agent import PPOagent
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def trainTRPO(PPO: PPOagent, episode_batch: int = 10, iterations: int = 100, graph=True):
    avgRewards = [PPO.nEpisodes(episode_batch) for _ in range(iterations)]
    if graph:
        plt.plot(range(iterations), avgRewards, linestyle='-', color='b', label='Rewards')
        plt.xlabel('Iteration number')
        plt.ylabel('Iteration Reward Achieved')
        plt.show()
    return avgRewards

def test(PPO: PPOagent, test_iterations: int = 20):
    avgRewards = PPO.nEpisodes(test_iterations)
    mean, std = np.mean(avgRewards), np.std(avgRewards)
    print(f'Test Mean: {mean: .2f}, Std: {std}')

if __name__ == '__main__':
    env = env = gym.make("Pendulum-v1")
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]

    policyAgent = Actor(state_dim, action_dim)
    valueAgent = Critic(state_dim)
    trpoAgent = PPOagent(env, policyAgent, valueAgent)
    trainTRPO(trpoAgent)
    test(trpoAgent)
