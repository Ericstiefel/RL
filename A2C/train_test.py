from a2c_agent import A2C_Agent
from AC import ActorCritic
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def runTraining(agent: A2C_Agent, env: gym.Env, iterations: int = 1000, n: int = 10, chart: bool = True):
    avg_episode_rewards = []
    for _ in range(1, iterations+1):
        avg_episode_rewards.append(agent.nEpisodes(env, n))

    if chart:
        plt.plot(range(iterations), avg_episode_rewards, linestyle='-', color='b', label='Rewards')
        plt.xlabel('Iteration number')
        plt.ylabel('Iteration Reward Achieved')
        plt.show()

def testing(agent: A2C_Agent, env: gym.Env, training_iter: int = 200, n: int = 10):
    rewards = []

    for _ in range(training_iter):
        rewards.append(agent.nEpisodes(env, n))

    mean, sd = np.mean(rewards), np.std(rewards)
    print(f'Eval complete with mean {mean: .2f}, with sd {sd: .2f}')
    return mean, sd


#Example Usage
env_name = 'CartPole-v1'
env = gym.make(env_name)
state_size, action_size = env.observation_space.shape[0], env.action_space.n
AC = ActorCritic(state_size, action_size)
ACagent = A2C_Agent(AC, add_entropy=True)
runTraining(ACagent, env)
testing(ACagent, env)


