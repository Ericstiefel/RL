import typing as tt
import numpy as np
import copy

State = np.ndarray
isValid = bool
Winner = int
GameOver = bool

class Board:
    def __init__(self, moveFirst: int):
        """
        Designed similar to the Gymnasium Environment
        White: 1
        Black: 2 
        Board dimensions: (6,7)

        Idea: for memory purposes store the entire state as a 42 bit int, but encoding decoding would slow down the search process
        """
        players = [1,2]
        assert(moveFirst in players), f'Non-available player chosen: {moveFirst}'
        self.toMove = moveFirst
        self.board = np.zeros((6, 7), dtype=int)
        self.validMoves = set(i for i in range(7))
    
    def reset(self):
        self.toMove = 1
        self.board = np.zeros((6, 7), dtype=int)
        self.validMoves = set(i for i in range(7))
    
    def __deepcopy__(self):
        new_board = Board(self.toMove)
        new_board.board = np.copy(self.board)
        new_board.validMoves = copy.deepcopy(self.validMoves)
        return new_board

    def dropPiece(self, col: int) -> bool:
        if not self.validMoves or col not in self.validMoves:
            return False
        for row in reversed(range(6)):
            if self.board[row][col] == 0:
                self.board[row][col] = self.toMove
                # If the column is full, remove it from validMoves
                if row == 0:
                    self.validMoves.remove(col)
                return True
        return False

    def check_winner(self, player: int) -> bool:
        # Check horizontal
        for r in range(6):
            for c in range(4):
                if all(self.board[r][c + i] == player for i in range(4)):
                    return True
        # Check vertical
        for r in range(3):
            for c in range(7):
                if all(self.board[r + i][c] == player for i in range(4)):
                    return True
        # Check diagonals (\)
        for r in range(3):
            for c in range(4):
                if all(self.board[r + i][c + i] == player for i in range(4)):
                    return True
        # Check diagonals (/)
        for r in range(3):
            for c in range(4):
                if all(self.board[r - i][c + i] == player for i in range(4)):
                    return True
        return False
    
    def print_board(self):
        for row in self.board:
            print("| " + " ".join(map(str, row)) + " |")
        print("  " + " ".join(map(str, range(7))))

    def place(self, col: int) -> tt.Tuple[isValid, Winner, GameOver]:
        
        is_valid = self.dropPiece(col) #To attach -inf as a reward for invalid move
        decisive = self.check_winner(self.toMove)
        over = decisive or len(self.validMoves) == 0
        if over:
            self.validMoves = set()
        winner = self.toMove if decisive else 0
        if is_valid:
            self.toMove = 3 - self.toMove  # Swap turns
        

        return is_valid, winner, over