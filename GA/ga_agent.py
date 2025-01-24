import numpy as np
import gymnasium as gym
from net import Net
class GAagent:
    def __init__(
            self, 
            env: gym.Env, 
            obs_size: int, 
            act_size: int, 
            ):
        self.env = env
        self.obs_size = obs_size
        self.act_size = act_size


    def nEpsiodes(self, net: Net, n: int = 10):
        totalReward = 0.0
        for _ in range(n):
            obs, _  = self.env.reset()
            while True:
                chosenAction = net.forward(obs)
                next_obs, reward, is_done, is_trunc, _ = self.env.step(chosenAction)
                totalReward += reward
                if is_done or is_trunc:
                    break
                obs = next_obs
        return totalReward / n
    
    def mutatePolicy(self, net: Net) -> Net:
        net_weights = net.weights  # List[layer1, layer2]
        for i in range(len(net_weights)):
            net_weights[i] += self.mutation_rate * np.random.randn(*net_weights[i].shape)
        return net

    
    def offspring(self, parent1: Net, parent2: Net):
        obs_size, actions = parent1.IOsize
        childNet = Net(obs_size, actions)
        for i in range(len(childNet.weights)):
            mask = np.random.randn(*childNet.weights[i].shape) > 0.5
            childNet.weights[i] = np.where(mask, parent1.weights[i], parent2.weights[i])
        return childNet

    def train(self, generations: int = 100, pop_size: int = 20, elite_pct: float = 0.2):
        best_net_val = []
        curr_population = [Net(self.obs_size, self.act_size) for _ in range(pop_size)]
        for generation in range(generations):
            fitness = np.array([self.nEpsiodes(net) for net in curr_population])
            sorted_indecies = np.argsort(-fitness)
            keep_top = int(pop_size * elite_pct)
            next_pop = [curr_population[i] for i in sorted_indecies[:keep_top]]
            while len(next_pop) < pop_size:
                parents = np.random.choice(next_pop, size=2, replace=False)
                child = self.offspring(parents[0], parents[1])
                next_pop.append(child)
            curr_population = next_pop
            best_net_val.append(fitness[sorted_indecies[0]])
        return curr_population[sorted_indecies[0]], best_net_val

