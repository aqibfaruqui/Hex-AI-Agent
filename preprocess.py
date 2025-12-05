import torch
import numpy as np


def encode_board(board, current_player):

    #create empty channels
    player = np.zeros((11, 11), dtype=np.float32)
    opponent = np.zeros((11, 11), dtype=np.float32)
    empty = np.zeros((11, 11), dtype=np.float32)

    #loop through every board position and assign it to the correct channel.
    for i in range(11):
        for j in range(11):
            val = board[i][j]

            #empty position
            if val == 0:
                empty[i, j] = 1.0
            #stone belongs to current player
            elif val == current_player:
                player[i, j] = 1.0
            #else opponent    
            else:
                opponent[i, j] = 1.0

    #stack the binary masks into one array with channels first  (3, 11, 11)
    stacked = np.stack([player, opponent, empty], axis=0)
    # Add a batch dimension so the network receives shape (1, 3, 11, 11)
    stacked = np.expand_dims(stacked, axis=0)
    #convert to tensor
    tensor = torch.from_numpy(stacked).float()

    return tensor

    

def move_to_device(tensor, device):
    #move tensor to correct device
    return tensor.to(device)