import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    Residual Block:
        Maintains the 'memory' of previous layers by adding the input (residual)
        back to the output after processing
    """
    def __init__(self, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class HexNN(nn.Module):
    """
    Residual Neural Network:
        Input: (B, 3, 11, 11)
        Channels:
            0 = current Player
            1 = opponent
            2 = empty cells
        Output:
            policy_logits: (B, 11, 11) - move probabilities
            value: (B,) - board evaluation (-1: Opponent win, +1: Player win)
    """
    def __init__(self, board_size=11, num_res_blocks=6, channels=64):
        super().__init__()
        self.board_size = board_size

        # Initial Convolutional Block
        self.start_block = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        # Residual Tower of 6 Blocks
        self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_res_blocks)])

        # Policy head (predicts next move probabilities)
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size)
        )

        # Value head (predicts who is winning)
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()       # Squeezes output to [-1, 1]
        )

    def forward(self, x):
        """
        Forward pass:
        x: tensor (B, 3, 11, 11)
        returns: policy_logits (B, 11, 11), value (B,)
        """
        # Shared trunk
        x = self.start_block(x)
        for res_block in self.blocks:
            x = res_block(x)

        # Policy head
        policy_logits = self.policy_head(x)
        policy_logits = policy_logits.view(-1, self.board_size, self.board_size)

        # Value head
        value = self.value_head(x).squeeze(-1)

        return policy_logits, value

def detect_gpu():
    """
    Detect torch device: GPU or CPU
    """

    if torch.cuda.is_available():
        print("NVIDIA GPU Detected")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():     # TODO: Remove and move to training on AP Calc's GPU
        print("Apple GPU Detected")
        return torch.device("mps")
    else:
        print("CPU Detected")
        return torch.device("cpu")

def load_model(weights_path: str = "agents/Group41/weights.pt"):
    """
    Load HexNN model and weights if file exists.

    Move model to correct device, and set eval mode.
    """
    device = detect_gpu()            
    model = HexNN()                 

    # Load weights if available
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded weights from {weights_path}")
    else:
        print("No weights found. Using random initialization.")

    model.to(device)
    model.eval()                     # evaluation mode that disables dropout/batchnorm
    return model


def evaluate_position(model, encoded_board):
    """
    Run forward pass on encoded board(s).
    encoded_board: tensor (B,3,11,11)
    returns: policy_logits (B,11,11), value (B,)
    """
    model.eval()  # ensure evaluation mode
    device = next(model.parameters()).device
    encoded_board = encoded_board.to(device)

    with torch.no_grad():  # no gradients needed for evaluation
        policy_logits, value = model(encoded_board)

    return policy_logits, value