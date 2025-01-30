from net import ConnectFourNet
from ConnectFour import Board
from mcts import MCTS

import random
import numpy as np
import torch
import torch.optim as optim
from collections import deque
import time

class Trainer:
    def __init__(
            self, 
            net: ConnectFourNet, 
            mcts: MCTS, 
              lr=0.001, 
              batch_size=32, 
              buffer_size=10000
              ):
        self.net = net
        self.mcts = mcts
        self.game = Board(1)
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=buffer_size)
        self.gamesPlayed = 0

    def self_play(self, games=50):
        for _ in range(games):
            self.game.reset()
            game_data = []
            while True:
                self.mcts.searchMiniBatches(self.game, self.net, batches=100)
                policy, _ = self.mcts.getPolicy(self.game, tau=1)
                game_data.append((self.game.board, policy))
                
                action = random.choices(range(7), weights=policy)[0]
                _, winner, game_over = self.game.place(action)
                
                if game_over:
                    value = 1 if winner == 1 else -1 if winner == 2 else 0
                    for s, p in game_data:
                        self.replay_buffer.append((s, p, value))
                    break
            self.gamesPlayed += 1

    def train(self, epochs=3):

        for epoch in range(epochs):
            if len(self.replay_buffer) < self.batch_size:
                continue 

            batch = random.sample(self.replay_buffer, self.batch_size)
            states, policies, values = zip(*batch)
            
            states = torch.tensor(np.array(states), dtype=torch.float32).unsqueeze(1) 
            policies = torch.tensor(np.array(policies), dtype=torch.float32)
            values = torch.tensor(np.array(values), dtype=torch.float32).unsqueeze(1)
            
            self.optimizer.zero_grad()
            pred_policies, pred_values = self.net(states)
            
            policy_loss = torch.nn.functional.cross_entropy(pred_policies, policies)
            value_loss = torch.nn.functional.mse_loss(pred_values, values)
            loss = policy_loss + value_loss
            
            loss.backward()
            self.optimizer.step()

net = ConnectFourNet()
mcts = MCTS()
model = Trainer(net, mcts)

#Timed training loop
startTime = time.time()
timeDiff = 0
while timeDiff < 5400: #In seconds
    model.self_play()
    model.train()
    timeDiff = time.time() - startTime
print('Games Played: ', model.gamesPlayed)

#Optional Save to use later without retraining
torch.save({
    'model_state_dict': net.state_dict(),
}, 'connect4net.pth')