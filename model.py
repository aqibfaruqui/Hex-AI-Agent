import torch
import torch.nn as nn


class SigmaHexNet(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: define layers for neural network

    def forward(self, x):
        # TODO: implement forward pass
        pass


def get_device():
    """
    return torch device (cuda/cpu)
    """
    # TODO: detect GPU and return device
    pass


def load_model(weights_path: str):
    """
    Create model, load weights, send to device, set eval mode.
    Return model.
    """
    # TODO: initialize model, load weights if exist, move to device
    # TODO: set model.eval()
    pass


def evaluate_position(model, encoded_board):
    """
    run forward pass on encoded board.
    encoded_board: (B, 3, 11, 11)
    returns model output.
    """
    # TODO: forward pass with no grad
    pass
