from policy_value import Policy, Value
from trpo_agent import TRPO
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def trainTRPO(TRPOagent: TRPO, episode_batch: int = 10, iterations: int = 1000, graph=True):
    avgRewards = []
    for _ in range(iterations):
        trajectories = TRPOagent.nEpisodes(episode_batch)
        avgRewards.extend([np.mean([rew for s, a, rew in trajectories])])
    if graph:
        plt.plot(range(iterations), avgRewards, linestyle='-', color='b', label='Rewards')
        plt.xlabel('Iteration number')
        plt.ylabel('Iteration Reward Achieved')
        plt.show()
    return avgRewards

def test(TRPOagent: TRPO, test_iterations: int = 100):
    trajectories = TRPOagent.nEpisodes(test_iterations)
    mean, std = np.mean([rew for s, a, rew in trajectories]), np.std([rew for s, a, rew in trajectories])
    print(f'Test Mean: {mean: .2f}, Std: {std}')

if __name__ == '__main__':
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]

    policyAgent = Policy(state_dim, action_dim)
    valueAgent = Value(state_dim)
    trpoAgent = TRPO(policyAgent, valueAgent, env)
    trainTRPO(trpoAgent)
    test(trpoAgent)
