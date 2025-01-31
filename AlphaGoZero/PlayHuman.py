import torch
from net import ConnectFourNet
from ConnectFour import Board

Net = ConnectFourNet()
checkpoint = torch.load('connect4net.pth') #Modify directory if necessary
Net.load_state_dict(checkpoint['model_state_dict'])  # Extract the state dictionary
Net.eval()


def playHuman(net: ConnectFourNet, iPlay: int):
    """
    Play a game of Connect Four against the neural network.
    iPlay: 1 for First (Human), 2 for Second (Human)
    """
    state = Board(1)  # Initialize the game board
    game_over = False

    while not game_over:
        
        if state.toMove == iPlay:
            # Human's turn
            state.print_board()
            print('Possible Moves: ', state.validMoves)
    
            action = int(input('Pick one of the available moves: '))
        else:
            # Neural network's turn
            print("Neural network is thinking...")
            board_tensor = torch.tensor(state.board, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            probs, _ = net.forward(board_tensor)
            action = torch.argmax(probs).item()  

        is_valid, winner, game_over = state.place(action)

        if not is_valid:
            print("Invalid move. Try again.")
            continue

        if game_over:
            state.print_board()
            if winner == iPlay:
                print("Congratulations! You win!")
            elif winner == 3 - iPlay:  
                print("Neural network wins!")
            else:
                print("It's a draw!")


playHuman(Net, iPlay=1)  