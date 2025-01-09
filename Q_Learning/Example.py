import gymnasium as gym 
from Q_Learning_Agent import qAgent
from train_test import train, test



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
