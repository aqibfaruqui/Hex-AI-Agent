import os
import time
import random
import torch
import numpy as np
from src.Colour import Colour
from src.Board import Board
from agents.Group41.preprocess import encode_board
from agents.Group41.mcts_cpp import BoardState, MCTS

# TODO: Add nice print statements for logging self play data generation

def run_self_play(games_per_batch=50, num_games=50):
    os.makedirs("agents/Group41/data", exist_ok=True)
    model_path = "agents/Group41/cpp_weights.pt"
    examples = []   # (board_state, policy_target, value_target)
    game_count = 0
    
    while game_count < num_games:
        start = time.time()
        board = BoardState(11)
        mcts = MCTS(board, model_path, True, 1.0)
        history = []    # (encoded_board, policy_probabilities, current_player)
        moves_count = 0

        while True:
            mcts.search(0.1)
            probs = np.array(mcts.get_action_probs(), dtype=np.float32)
            prob_sum = np.sum(probs)
            
            # Normalise exactly to 1.0
            if prob_sum > 0:
                probs /= prob_sum

            policy_target = np.array(probs, dtype=np.float32).reshape(11, 11)   # convert the policy list to numpy array
            raw_board_list = board.board
            raw_board_np = np.array(raw_board_list, dtype=np.int8).reshape(11, 11)

            player = board.current_colour
            encoded_board = encode_board(raw_board_np, player).cpu()   # Move training data to CPU to save GPU for inference
            history.append([encoded_board, policy_target, player])

            # Pick move & play
            move_idx = np.random.choice(121, p=probs)
            board.play_move(move_idx)
            mcts.update_root(move_idx)
            moves_count += 1

            # Play to end of game
            value, finished = board.get_result()
            if finished:
                winner = board.winner
                # Backpropagate outcome
                for hist_board, hist_policy, hist_player in history:
                    outcome = 1.0 if hist_player == winner else -1.0
                    examples.append((hist_board, hist_policy, outcome))

                game_count += 1
                winner_name = "RED" if winner == 1 else "BLUE"
                print(f"Game {game_count} finished in {time.time()-start:.2f}s. Moves: {moves_count}. Winner: {winner_name}")
                break

        # Saving generated data files in batches for neural network training
        if len(examples) >= (games_per_batch * 50): # Assuming 50 moves per game
            timestamp = int(time.time())
            filename = f"agents/Group41/data/data_cpp_{timestamp}.pt"
            torch.save(examples, filename)
            print(f'Saved batch to {filename} ({len(examples)} positions)')
            examples = []   # Reset buffer

if __name__ == "__main__":
    run_self_play()
