import torch
import numpy as np

def encode_board(board, current_player):
    """
    Optimized board encoding using NumPy vectorization.
    Input:
        board: (11, 11) numpy array
        current_player: int (1 or 2)
    Output:
        Tensor shape (1, 3, 11, 11)
    """
    opponent = 3 - current_player
    
    # NumPy creates 3 boolean arrays
    mask_player = (board == current_player)
    mask_opponent = (board == opponent)
    mask_empty = (board == 0)
    
    # Stack and convert to Float (True -> 1.0, False -> 0.0)
    stacked = np.stack([mask_player, mask_opponent, mask_empty], axis=0).astype(np.float32)
    
    # Add Batch Dimension and convert to Tensor
    # Output shape: (1, 3, 11, 11)
    return torch.from_numpy(stacked).unsqueeze(0)
    
def move_to_device(tensor, device):
    return tensor.to(device)