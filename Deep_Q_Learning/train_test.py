from Processing_Images import AtariWrapper
from Conv import CNN
from Deep_Learning_Agent import DQNAgent
import gymnasium as gym
import numpy as np

#For specific Atari Environment
from ale_py import ALEInterface
import ale_py
import shimmy


def runTraining(agent: DQNAgent, env: gym.Env, training_iterations=500, avg_over=10):
    for episode in range(1, training_iterations+1):
        avg_rew = 0.0
        for _ in range(avg_over):
            avg_rew += agent.play_episode(env) / avg_over
        agent.train()
        print(f"Episode {episode}, Average Iteration Reward: {avg_rew}")
    return agent

def test(agent: DQNAgent, env: gym.Env, test_iter=100):
    rewards = []
    
    for _ in range(test_iter):
        rewards.append(agent.play_episode(env))
    mean, sd = np.mean(rewards), np.std(rewards)

    print(f'Testing Results: Avg {mean: .2f} +- {sd: .2f}')
    return mean, sd

#Training Example

env = gym.make('ALE/SpaceInvaders-v5', render_mode='rgb_array')
wrapped_env = AtariWrapper(env, stack_size=4, new_shape=(84, 84))
agent = DQNAgent(
    state_shape=(84, 84, 4),
    action_dim=wrapped_env.action_space.n,
)

trainedAgent = runTraining(agent, wrapped_env)
test(trainedAgent, wrapped_env)