import os
import sys
import torch
import numpy as np

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

try:
    from agents.Group41.mcts_cpp import BoardState, MCTS
except ImportError:
    print("WARNING: C++ extension not found. Agent will fail unless compiled.")


class HexAgent(AgentBase):
    """This class implements our Hex agent.

    The class inherits from AgentBase, which is an abstract class.
    The AgentBase contains the colour property which you can use to get the agent's colour.
    You must implement the make_move method to make the agent functional.
    You CANNOT modify the AgentBase class, otherwise your agent might not function.
    """

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self_dir, "cpp_weights.pt")
        
        if not os.path.exists(self.model_path):
            print(f"CRITICAL ERROR: Model weights not found at {self.model_path}")        
        self.use_gpu = torch.cuda.is_available()

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """The game engine will call this method to request a move from the agent.
        If the agent is to make the first move, opp_move will be None.
        If the opponent has made a move, opp_move will contain the opponent's move.
        If the opponent has made a swap move, opp_move will contain a Move object with x=-1 and y=-1,
        the game engine will also change your colour to the opponent colour.

        Args:
            turn (int): The current turn
            board (Board): The current board state
            opp_move (Move | None): The opponent's last move

        Returns:
            Move: The agent's move
        """

        # Handle Swap Rule
        # TODO: Add opening move database
        if turn == 2:
            return Move(-1, -1)

        tiles = board.tiles
        flat_board = []
        for row in range(11):
            for col in range(11):
                tile = tiles[row][col]
                if tile.colour == Colour.RED:
                    val = 1
                elif tile.colour == Colour.BLUE:
                    val = 2
                else:
                    val = 0
                flat_board.append(val)

        internal_board = BoardState(11)
        internal_board.set_board_from_vector(flat_board)

        my_colour_int = 1 if self.colour == Colour.RED else 2
        internal_board.current_colour = my_colour_int
        
        mcts = MCTS(internal_board, self.model_path, self.use_gpu, 1.0)
        best_move_int = mcts.search(4.9)

        x = best_move_int // 11
        y = best_move_int % 11
        
        return Move(x, y)
