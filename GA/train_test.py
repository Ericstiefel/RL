import gymnasium as gym
from ga_agent import GAagent
import matplotlib.pyplot as plt

GENERATIONS = 1000
POP_SIZE = 50
env = gym.make('CartPole-v1')
obs_size, action_size = env.observation_space.shape[0], env.action_space.n
#Train
agent = GAagent(env, obs_size, action_size)

net, best_rewards = agent.train(generations=GENERATIONS, pop_size=POP_SIZE)

#graph
plt.plot(range(GENERATIONS), best_rewards, linestyle='-', color='b', label='Rewards')
plt.xlabel('Iteration number')
plt.ylabel('Iteration Reward Achieved')
plt.show()

#Test
avg_rew = agent.nEpsiodes(net, 500)
print('Average Rewards: ', avg_rew)