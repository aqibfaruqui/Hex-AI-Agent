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
        self.model = load_model("agents/Group41/cpp_weights.pt")
        self.use_gpu = torch.cuda.is_available()
        self.internal_board = BoardState(11)

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

        raw_grid = board.board
        
        flat_board = []
        for row in range(11):
            for col in range(11):
                val = raw_grid[row][col]    # 0=Empty, 1=Red, 2=Blue
                flat_board.append(int(val))

        self.internal_board.set_board_from_vector(flat_board)
        my_colour_int = 1 if self.colour == Colour.RED else 2
        self.internal_board.current_colour = my_colour_int
        mcts = MCTS(self.internal_board, self.model_path, self.use_gpu, 1.0)
        best_move_int = mcts.search(4.9)

        x = best_move_int // 11
        y = best_move_int % 11
        
        return Move(x, y)
