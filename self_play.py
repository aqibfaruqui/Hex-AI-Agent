import torch
import numpy as np
import os
import time
from src.Colour import Colour
from src.Board import Board
from agents.Group41.board_state import BoardStateNP
from agents.Group41.mcts import MCTS
from agents.Group41.model import load_model, HexNN
from agents.Group41.preprocess import encode_board

# TODO: Add nice print statements for logging self play data generation

def run_self_play(games_per_batch=1, num_games=10):
    os.makedirs("agents/Group41/data", exist_ok=True)
    model = load_model("agents/Group41/weights.pt")     # Load current best model
    model.eval()
    examples = []   # (board_state, policy_target, value_target)
    game_count = 0
    
    while game_count < num_games:
        start = time.time()
        board = BoardStateNP(Board(11))
        mcts = MCTS(game=None, model=model, tt=True)
        history = []    # (encoded_board, policy_probabilities, current_player)
        moves_count = 0

        while True:
            # Get MCTS root node to extract probabilities (as opposed to a 'best move')
            root = mcts.search_for_selfplay(board, time_limit=0.5)

            # Extract target policy (search probabilities)
            # Network tries to predict 'visit count / total visits' for each child
            counts = np.zeros((11, 11), dtype=np.float32)
            total_visits = sum(child.visits for child in root.children.values())

            if total_visits > 0:
                for move, child in root.children.items():
                    counts[move.x, move.y] = child.visits
                policy_target = counts / total_visits
            else:
                policy_target = np.ones((11, 11)) / 121.0       # TODO: Confirm that this shouldn't be possible

            # Save board state before 'self playing' a move
            player = 1 if board.current_colour == Colour.RED else 2
            encoded_board = encode_board(board.get_numpy(), player).cpu()   # Move training data to CPU to save GPU for inference
            history.append([encoded_board, policy_target, board.current_colour])

            # Select Best Move (Deterministic for strong play, or Weighted for exploration?)
            # AlphaZero uses weighted random for first 30 moves, then deterministic.
            # For simplicity now: Weighted Random (Exploration) that MCTS would have played
            # best_move = weighted_pick(root)   # TODO: Implement later for better exploration
            
            # best_move = root.most_visits()    # TODO: Or allow this to play again when there is some trained data
            # board.play_move(best_move)

            # For now, temperature sampling to randomly pick based on visit count 
            # (instead of defaulting to max visits) to see different games even
            # on an untrained neural network
            moves = list(root.children.keys())
            visits = [child.visits for child in root.children.values()]
            if visits:
                probs = np.array(visits) / sum(visits)
                choice_idx = np.random.choice(len(moves), p=probs)
                best_move = moves[choice_idx]
            else:
                # Fallback if no simulations (shouldn't happen)
                best_move = random.choice(board.get_legal_moves())
            board.play_move(best_move)
            moves_count += 1

            # Play to end of game
            value, finished = board.get_result()
            if finished:
                winner = board._opponent(board.current_colour)  # New current player must be the loser if game just finished

                # Backpropagate result
                for _board, _policy, _player in history:
                    outcome = 1.0 if _player == winner else -1.0
                    examples.append((_board, _policy, outcome))

                game_count += 1
                print(f'Game {game_count} finished in {time.time()-start:.1f}s. Winner: {winner}')
                break

        # Saving generated data files in batches for neural network training
        if len(examples) >= (games_per_batch * 50): # Assuming 50 moves per game
            timestamp = int(time.time())
            filename = f"agents/Group41/data/data_{timestamp}.pt"
            torch.save(examples, filename)
            print(f'Saved batch to {filename} ({len(examples)} positions)')
            examples = []   # Reset buffer

if __name__ == "__main__":
    run_self_play()
