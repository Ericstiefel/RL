from net import Net
from ES_Agent import ESagent
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import time



def trainESagent(ESag: ESagent, iterations: int, graph: bool = True):
    avgRewards = np.mean([ESag.trainStep() for _ in range(iterations)], axis=1)

    if graph:
        plt.plot(range(iterations), avgRewards, linestyle='-', color='b', label='Rewards')
        plt.xlabel('Iteration number')
        plt.ylabel('Iteration Reward Achieved')
        plt.show()
    return avgRewards

def test(ESag: ESagent, test_iterations: int = 1000):
    startTime = time.time()
    avgRewards = ESag.nEpisodes(test_iterations)
    endTime = time.time()
    timeDiff = endTime - startTime

    mean, std = np.mean(avgRewards), np.std(avgRewards)
    print(f'Test Mean: {mean: .2f}, Std: {std}')
    print('Time: ', timeDiff)
    return mean, std

env = gym.make('CartPole-v1')
obs_dim, action_dim = env.observation_space.shape[0], env.action_space.n
net = Net(obs_dim, action_dim)
ESag = ESagent(net, env)

trainESagent(ESag, 10000)
test(ESag)