import gymnasium as gym
import numpy as np
import typing as tt
from collections import defaultdict
import random

#define new types
State = int
Action = int
ValueKey = tt.Tuple[State, Action]

class qAgent:
  def __init__(self, alpha=0.7, gamma=0.95, epsilon=1.0, decay=0.9995):
      self.Q: tt.Dict[ValueKey, float] = defaultdict(float)
      self.epsilon = epsilon
      self.alpha = alpha
      self.gamma = gamma
      self.decay = decay

  def bestAction(self, env: gym.Env, state: State) -> tt.Tuple[Action, float]:
      bestVal = -float('inf')  
      bestAct = 0

      for action in range(env.action_space.n):
          currentVal = self.Q[(state, action)]
          if currentVal > bestVal:
              bestAct, bestVal = action, currentVal
      return bestAct, bestVal

  def valueUpdate(self, env: gym.Env, state: State, action: Action, reward: float, next_state: State) -> None:
    _, max_next_val = self.bestAction(env, next_state)
    key = (state, action)
    self.Q[key] += self.alpha * (reward + self.gamma * max_next_val - self.Q[key]) #Bellman Value Update

  def playEpisode(self, env: gym.Env) -> float:
    assert isinstance(env, gym.Env)

    state, _ = env.reset()
    eps_reward = 0.0

    while True:
        if random.random() < self.epsilon:
            action = env.action_space.sample()
            exRew = 0
            
        else:
            action, exRew = self.bestAction(env, state)  

        next_state, reward, done, _, _ = env.step(action)
        eps_reward += reward

        self.valueUpdate(env, state, action, reward, next_state)
        state = next_state

        if done:
            break

    self.epsilon = max(0.01, self.epsilon * self.decay)
    return eps_reward
