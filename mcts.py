import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from typing import List, Tuple

from src.Move import Move
from src.Colour import Colour
from agents.Group41.preprocess import encode_board
from agents.Group41.board_state import BoardStateNP

class Node:
    def __init__(self, parent=None, prior=0):
        self.parent = parent
        self.children = {}      # Move -> Node
        self.visits = 0         # N
        self.total_value = 0    # W (total value)
        self.prior = prior      # P (probability from neural network)

    @property
    def value(self) -> int:
        """Returns Q (Expected win rate)"""
        return self.total_value / self.visits if self.visits != 0 else 0

    def is_leaf(self) -> bool:
        """A node with no children is a leaf node"""
        return len(self.children) == 0

    def expand(self, priors: List[Tuple[Move, int]]) -> None:
        """
        Expands node by creating children
        priors: List of (move, prob) tuples from neural network
        """
        for move, prob in priors:
            if move not in self.children:
                self.children[move] = Node(parent=self, prior=prob)

    def update(self, value: int) -> None:
        """Updates node stats during backpropagation"""
        self.visits += 1
        self.total_value += value

    def most_visits(self) -> Move:
        """Returns move with highest visit count"""
        return max(self.children.items(), key=lambda item: item[1].visits)[0]

class MCTS:
    def __init__(self, game, model, tt=True):
        self.game = game
        self.model = model
        self.c_puct = 1.0       # higher c_puct = higher tendency to explore 'new' knowledge in find_best_child()
        self.tt = tt            # optionally use transposition table for experiment
        self.transposition = {}
        self.transposition_hits = 0
        self.total_simulations = 0

    def search(self, board: BoardStateNP, time_limit: float) -> Move:
        """
        Main MCTS Loop

        Args:
            board (Board): The current board state
            time_limit (int): Time allowed for move in seconds
        """
        root = self._run_search(board, time_limit)

        # TODO: Clarify if root may have no children for small time_limit
        if not root.children:
            return random.choice(board.get_legal_moves())
            
        # TODO: Explore using action = N(s,a)^(1/t) / Σ_b N(s,b)^(1/t) and finetuning 't'
        return root.most_visits()

    def search_for_selfplay(self, board: BoardStateNP, time_limit: float) -> Move:
        """
        Runs MCTS and returns root node (for data extraction)
        Identical logic to search(), just different return type :D
        """
        return self._run_search(board, time_limit)

    def find_best_child(self, node: Node) -> Tuple[Move, Node]:
        """Choose child with highest PUCT"""
        best_score = float('-inf')
        best_move = None
        best_child = None

        sqrt_parent_visits = math.sqrt(node.visits)

        for move, child in node.children.items():
            q = child.value
            u = self.c_puct * child.prior * (sqrt_parent_visits / (1 + child.visits))
            score = q + u

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def _run_search(self, board: BoardStateNP, time_limit: float) -> Node:
        """Private helper of MCTS main loop to support runtime and selfplay data extraction"""
        root = Node()
        start = time()
        self.transposition_hits = 0
        self.total_simulations = 0

        while time() - start < time_limit:
            self.total_simulations += 1

            node = root
            search_board = board.copy()

            # 1: Selection
            while not node.is_leaf():
                move, node = self.find_best_child(node)
                search_board.play_move(move)

                if self.tt:
                    board_bytes = search_board.get_numpy().tobytes()
                    if board_bytes in self.transposition and self.transposition[board_bytes] is not node:
                        node = self.transposition[board_bytes]
                        self.transposition_hits += 1
            
            # 2: Expansion
            value, finished = search_board.get_result()

            if not finished:
                if self.model:
                    # Preparing neural network input
                    player = 1 if search_board.current_colour == Colour.RED else 2
                    input_tensor = encode_board(search_board.get_numpy(), player)
                    device = next(self.model.parameters()).device

                    # Inference
                    with torch.no_grad():
                        policy_vector, value_tensor = self.model(input_tensor)

                    # Value is expected win rate [-1, 1]
                    value = value_tensor.item()

                    # Process policy
                    legal_moves = search_board.get_legal_moves()
                    logits = policy_vector.squeeze() # (11, 11)
                    move_logits = []
                    for move in legal_moves:
                        move_logits.append(logits[move.x, move.y])
                    
                    # Ensuring sum of probabilities = 1.0
                    move_probs = F.softmax(torch.stack(move_logits), dim=0).tolist()

                    # Expand
                    priors = list(zip(legal_moves, move_probs))
                    node.expand(priors)
                
                else:
                    value = self.random_simulation(search_board)
                    legal_moves = search_board.get_legal_moves()
                    priors = [(m, 1.0/len(legal_moves)) for m in legal_moves]
                    node.expand(priors)

            if self.tt:
                board_bytes = search_board.get_numpy().tobytes()
                self.transposition[board_bytes] = node

            # 3: Backpropagation
            while node is not None:
                node.update(value)
                value = -value
                node = node.parent
        
        return root

    def random_simulation(self, board: BoardStateNP) -> int:
        """Private helper of MCTS main loop to support development without completed neural network"""
        search_board = board.copy()

        while True:
            value, finished = search_board.get_result()
            if finished:
                return value

            moves = search_board.get_legal_moves()
            if not moves: break
            move = random.choice(moves)
            search_board.play_move(move)
        
        return 0
                