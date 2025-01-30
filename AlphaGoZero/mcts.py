import typing as tt
from ConnectFour import Board
import math
from net import ConnectFourNet
import torch
import numpy as np

State = np.ndarray
isValid = bool
Winner = int
GameOver = bool
Reward = float
Action = int

class MCTS:
    def __init__(self, c_puct: float = 1.0):
        self.c_puct = c_puct
        self.visit_count = {} #N(s,a)
        self.value = {} #Total Value
        self.avg_value = {} #Q(s,a)
        self.probs = {} #P(s,a)

    def clear(self): 
        self.visit_count.clear() 
        self.value.clear() 
        self.avg_value.clear() 
        self.probs.clear()
    

    def findLeaf(self, state: Board) -> tt.Tuple[Reward, State, list[State], list[Action]]:
        player = state.toMove
        states = []
        actions = []
        curr_state: Board = state
        reward = 0

        while True:
            curr_board = curr_state.board.tobytes()
            if curr_board not in self.visit_count:
                self.visit_count[curr_board] = np.zeros(7, dtype=int)
                self.value[curr_board] = np.zeros(7, dtype=float)
                self.avg_value[curr_board] = np.zeros(7, dtype=float)
                self.probs[curr_board] = np.ones(7) / 7 

            counts = self.visit_count[curr_board]
            visits_sqrt = math.sqrt(sum(counts))
            probs = self.probs[curr_board]
            values_avg = self.avg_value[curr_board]

            if curr_state.board.all() == state.board.all():
                noises = np.random.dirichlet([0.03], 7) #Inject Noise into Action Space
                probs = [0.75 * p + 0.25 * n for p, n in zip(probs, noises)]
            score = [
                v + self.c_puct * p * visits_sqrt / (1 + count)
                for v, p, count in zip(values_avg, probs, counts)
            ]

            invalid_actions = set(i for i in range(7)) - curr_state.validMoves
            for inv_act in invalid_actions:
                score[inv_act] = np.array([-np.inf])
            action = np.argmax(score)
            isValid, winner, GameOver =  curr_state.place(action)
            actions.append(action)
            states.append(curr_board)

            reward = 0 if isValid else float('-inf')

            if GameOver:
                if winner:
                    reward = 1 if player == winner else -1
                else:
                    reward = 0.5

                break

        return reward, curr_state, states, actions
    
    def searchMiniBatches(self, state: Board, net: ConnectFourNet, batches: int):
        
        for _ in range(batches):
            copyState = state.__deepcopy__()

            backup_queue = []
            expand_states = []
            expand_queue = []
            planned = set()

            reward, leaf_state, boards, actions = self.findLeaf(copyState)
            
            if reward != 0:  # Terminal state, backup immediately
                backup_queue.append((reward, boards, actions))
            else:
                if leaf_state not in planned:
                    planned.add(leaf_state)
                    expand_states.append(leaf_state)
                    expand_queue.append((leaf_state, boards, actions))

            if expand_queue:
                state_tensors = np.array([s for s, _, _ in expand_queue])  # Convert board states to NN input
                state_tensors = torch.tensor(state_tensors, dtype=torch.int8)
                probs, values = net.forward(state_tensors)  # Predict policy and value
                
                for (leaf_state, boards, actions), prob, value in zip(expand_queue, probs, values):
                    self.visit_count[leaf_state] = np.zeros(7, dtype=int)
                    self.value[leaf_state] = np.zeros(7, dtype=float)
                    self.avg_value[leaf_state] = np.zeros(7, dtype=float)
                    self.probs[leaf_state] = prob  # Store NN action probabilities
                    
                    backup_queue.append((value, boards, actions))

            # Update Values
            for reward, states, actions in backup_queue:
                for s, a in zip(states, actions):
                    self.visit_count[s][a] += 1
                    self.value[s][a] += reward
                    self.avg_value[s][a] = self.value[s][a] / self.visit_count[s][a]

    def getPolicy(self, state: Board, tau=1):
        curr_board = state.board.tobytes()
        if curr_board not in self.visit_count:
            self.visit_count[curr_board] = np.zeros(7, dtype=int)
            self.value[curr_board] = np.zeros(7, dtype=float)
            self.avg_value[curr_board] = np.zeros(7, dtype=float)
            self.probs[curr_board] = np.ones(7) / 7  

        counts = self.visit_count[curr_board]
        if tau == 0:  # Becomes deterministic
            probs = np.zeros(7, dtype=float)
            probs[np.argmax(counts)] = 1.0
        else:
            counts = np.array(counts, dtype=float)  
            counts = counts ** (1 / tau)
            total = np.sum(counts)

            if total == 0:  
                probs = np.ones(7, dtype=float) / 7
            else:
                probs = counts / total 

        
        probs = np.array(probs, dtype=float).flatten()

        probs /= np.sum(probs)

        return probs, self.avg_value[curr_board]
