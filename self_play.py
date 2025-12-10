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

# TODO: Add nice print statements for logging self play data generation

def run_self_play(num_games=100, filepath="agents/Group41/data.pt"):
    # Load current best model
    model = load_model("agents/Group41/weights.pt")
    examples = []   # (board_state, policy_target, value_target)

    for i in range(num_games):
        start = time.time()
        board = BoardStateNP(Board(11))
        mcts = MCTS(game=None, model=model)
        history = []    # (encoded_board, policy_probabilities, current_player)

        while True:
            # Get MCTS root node to extract probabilities (as opposed to a 'best move')
            root = mcts.search_for_selfplay(board, time_limit=0.5)

            # Extract target policy (search probabilities)
            # Network tries to predict 'visit count / total visits' for each child
            counts = np.zeros((11, 11), dtype=np.float32)
            visits = sum(child.visits for child in root.children.values())

            for move, child in root.children.items():
                counts[move.x, move.y] = child.visits

            if visits > 0:
                policy_target = counts / visits
            else:
                policy_target = np.ones((11, 11)) / 121.0       # TODO: Confirm that this shouldn't be possible

            # Save board state before 'self playing' a move
            player = 1 if board.current_colour == Colour.RED else 2
            encoded_board = encode_board(board.get_numpy(), player)
            history.append([encoded_board, policy_target, board.current_colour])

            # Play the move MCTS would have picked
            best_move = roots.most_visits()
            board.play_move(best_move)

            # Play to end of game
            value, finished = board.get_result()
            if finished:
                winner = board._opponent(board.current_colour)  # New current player must be the loser if game just finished

                # Backpropagate result
                for _board, _policy, _player in history:
                    outcome = 1.0 if _player == winner else -1.0
                    examples.append((_board, _policy, outcome))

                break

        # Saving generated data files in batches for neural network training
        id = int(time.time())
        torch.save(examples, f"agents/Group41/data_{id}.pt")

if __name__ == "__main__":
    run_self_play()
