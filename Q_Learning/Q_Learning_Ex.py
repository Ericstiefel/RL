from Q_Learning_Agent import qAgent
import gymnasium as gym
import numpy as np

#Examples of the Q-Learning Agents in both the gymnasium FrozenLake and Taxi environments
#Documentation and explanation of the environments can be found at https://gymnasium.farama.org/index.html

def train(map: gym.Env, agent: qAgent, batch_iter=10, max_iter=10000, solved_bound=0.95):
  best_reward = float('-inf')
  iter_no = 0
  while iter_no <= 100000:
    iter_no += 1
    
    curr_reward = 0.0
    for _ in range(batch_iter):
      curr_reward += agent.playEpisode(map) / batch_iter
      
    if curr_reward > best_reward:
      print(f'Updated Reward: {best_reward: .2f} -> {curr_reward: .2f} on iteration {iter_no}')
      best_reward = curr_reward
    if curr_reward >= solved_bound:
      print('Solved with reward of', curr_reward)
      break

def test(map: gym.Env, agent: qAgent, testing_iter=100):
  iter_no = 0
  rewards = []
  testingEnv = map

  while iter_no <= testing_iter:
    iter_no += 1
    rewards.append(agent.playEpisode(testingEnv))
    
  mean, sd = np.mean(rewards), np.std(rewards)

  print(f"Mean_reward={mean:.2f} +/- {sd:.2f}")
  return mean, sd

# Train and test FrozenLake agent
FROZEN_LAKE_ENV = 'FrozenLake-v1'
fl_env = gym.make(FROZEN_LAKE_ENV, render_mode="rgb_array")
fl_agent = qAgent()
train(fl_env, fl_agent, solved_bound=0.95, batch_iter=30)
test(fl_env, fl_agent)

TAXI_ENV = 'Taxi-v3'
# Train and test Taxi agent
taxi_env = gym.make(TAXI_ENV, render_mode="rgb_array")
taxi_agent = qAgent()
train(taxi_env, taxi_agent, solved_bound=9.7, batch_iter=30)
test(taxi_env, taxi_agent)
