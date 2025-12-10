import torch
import numpy as np
import os
import time
from src.Colour import Colour
from src.Board import Board
from agents.Group41.board_state import BoardStateNP
from agents.Group41.mcts import MCTS
from agents.Group41.model import load_model
from agents.Group41.preprocess import encode_board

def run_self_play(num_games=100, filepath="agents/Group41/data.pt"):
    # Load current best model
    model = load_model("agents/Group41/weights.pt")
    examples = []   # (board_state, policy_target, value_target)

    for i in range(num_games):
        start = time()
        board = BoardStateNP(Board(11))
        mcts = MCTS(game=None, model=model)
        history = []    # (encoded_board, policy_probabilities, current_player)

        while True:
            
            # TODO: Simulate game playing with MCTS

            value, finished = board.get_result()
            if finished:
                # TODO: Backpropagate result
                break

        # Saving generated data files in batches for neural network training
        id = int(time())
        torch.save(examples, f"agents/Group41/data_{id}.pt")

if __name__ == "__main__":
    run_self_play()
