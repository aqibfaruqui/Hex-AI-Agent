import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import glob
import os
import random
from agents.Group41.model import HexNN, load_model

BATCH_SIZE = 64        # Adjust based on GPU VRAM (maybe 64-256)
LEARNING_RATE = 0.001
EPOCHS = 1             # Train 1 epoch on the new data
WEIGHTS_PATH = "agents/Group41/weights.pt"

class HexDataset(Dataset):
    def __init__(self, data_files):
        self.samples = []
        for file in data_files:
            try:
                data = torch.load(file)
                self.samples.extend(data)
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        board, policy, value = self.samples[idx]
        # board is already tensor (3,11,11)
        # policy is numpy (11,11) -> needs flattening to 121
        # value is float -> needs tensor
        
        policy_tensor = torch.tensor(policy.flatten(), dtype=torch.float32)
        value_tensor = torch.tensor(value, dtype=torch.float32)
        
        return board, policy_tensor, value_tensor

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    if os.path.exists(WEIGHTS_PATH):
        model = load_model(WEIGHTS_PATH)
    else:
        model = HexNN().to(device)
    
    model.train() # Switch to 'train' mode to enable Dropout/BatchNorm updates (which 'eval' doesn't have)
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    files = glob.glob("agents/Group41/data_*.pt")
    if not files:
        print("No training data found! Run selfplay.py first :D")
        return

    print(f"Found {len(files)} data files.")
    dataset = HexDataset(files)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Dataset size: {len(dataset)} positions")

    total_loss = 0
    steps = 0
    
    # Loss functions
    # Policy: Cross Entropy (Probability distribution)
    # Value: MSE (Mean Squared Error for regression -1 to 1)
    mse_loss = nn.MSELoss() 
    
    # Copying buffers from CPU to GPU
    for batch_idx, (boards, target_policy, target_value) in enumerate(loader):
        boards, target_policy, target_value = boards.to(device), target_policy.to(device), target_value.to(device)

        optimiser.zero_grad()   # Resetting gradient buffers for new batch
        pred_policy, pred_value = model(boards)   # Forward pass
        
        # Reshape pred_policy from (B, 11, 11) to (B, 121) to match target
        pred_policy_flat = pred_policy.view(pred_policy.size(0), -1)

        # Calculate Loss
        # Policy Loss: LogSoftmax + NLLLoss (or simpler: CrossEntropy between distributions)
        # We manually implement CrossEntropy for soft targets: -sum(target * log(pred))
        log_probs = F.log_softmax(pred_policy_flat, dim=1)
        policy_loss = -torch.sum(target_policy * log_probs) / target_policy.size(0)
        value_loss = mse_loss(pred_value, target_value)
        
        loss = policy_loss + value_loss
        loss.backward()
        optimiser.step()            # i.e. weight = weight - (learning rate * gradient)
        total_loss += loss.item()   # Running sum of error scores
        steps += 1
        
        # Printing loss (should go down as we train)
        if batch_idx % 10 == 0:
            print(f"Step {batch_idx}/{len(loader)} | Loss: {loss.item():.4f} (P: {policy_loss.item():.4f}, V: {value_loss.item():.4f})")

    # 4. Save
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Training Complete. Weights saved to {WEIGHTS_PATH}")
    
    # TODO: Cleanup - Delete processed files to prevent re-training on same data forever?
    # For now keep and later move to an 'archive' folder

if __name__ == "__main__":
    train()