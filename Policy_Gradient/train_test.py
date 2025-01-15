import gymnasium as gym
from reinforce import Reinforce
import numpy as np

def runTraining(agent: Reinforce, env: gym.Env, episode_num: int = 2000, solved_bound = float('inf'), consistency: int = 20):
    streak = 0
    for episode_num in range(1, episode_num+1):
        reward = agent.playEpisode(env)
        if reward >= solved_bound:
            streak += 1
        else:
            streak = 0        
        if streak == consistency:
            print('Solved!')
            break
        

def testing(agent: Reinforce, env: gym.Env, training_iter: int = 200):
    rewards = []

    for _ in range(training_iter):
        rewards.append(agent.playEpisode(env))

    mean, sd = np.mean(rewards), np.std(rewards)
    print(f'Eval complete with mean {mean: .2f}, with sd {sd: .2f}')
    return mean, sd


#Example usage
env_name = 'CartPole-v1'
env = gym.make(env_name)
state_dim, action_dim = env.observation_space.shape[0], env.action_space.n

agent = Reinforce(state_dim, action_dim)
runTraining(agent, env)
testing(agent, env)
