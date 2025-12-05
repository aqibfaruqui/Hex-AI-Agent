import torch
import numpy as np


def encode_board(board, current_player):
    """
    convert board state into tensor

    board: 2D list or array
    current_player: 1 or -1

    return FloatTensor
    """
    # TODO: convert board to channels
    # TODO: stack channels into tensor
    # TODO: add batch dimension
    pass


def move_to_device(tensor, device):
    """
    move tensor to correct device (cpu/cuda)
    """
    # TODO: send tensor to device
    pass
