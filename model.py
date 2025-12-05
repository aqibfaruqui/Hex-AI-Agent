import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class HexNN(nn.Module):
    """
    Neural Network:
        Input: (B, 3, 11, 11)
        Channels:
            0 = current Player
            1 = opponent
            2 = empty cells
        Output:
            policy_logits: (B, 11, 11) - move probabilities
            value: (B,) - board evaluation (-1: Opponent win, +1: Player win)
    """
    def _init_(self, board_size=11, channels=64):
        super()._init_()
        self.board_size = board_size

        # Shared convolutional trunk
        self.conv1 = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        # Policy head (predicts next move probabilities)
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)  # 2 channels for policy
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # Value head (predicts who is winning)
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """
        Forward pass:
        x: tensor (B, 3, 11, 11)
        returns: policy_logits (B, 11, 11), value (B,)
        """
        # Shared trunk
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Policy head
        p = F.relu(self.policy_conv(x))  # output: (B,2,11,11)
        p = p.view(p.size(0), -1)        # flatten to (B, 2*11*11)
        policy_logits = self.policy_fc(p)
        policy_logits = policy_logits.view(-1, self.board_size, self.board_size)

        # Value head
        v = F.relu(self.value_conv(x))   # output: (B,1,11,11)
        v = v.view(v.size(0), -1)        # flatten to (B, 11*11)
        v = F.relu(self.value_fc1(v))    # hidden layer
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)  # output in [-1,1]

        return policy_logits, value


def detect_gpu():
    """
    Detect torch device: GPU or CPU
    """

    if torch.cuda.is_available():
        print("GPU Detected")
        return torch.device("cuda")
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