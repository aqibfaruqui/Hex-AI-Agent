from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

from agents.Group41.mcts import MCTS
from agents.Group41.board_state import BoardStateNP
from agents.Group41.model import load_model

class HexAgent(AgentBase):
    """This class implements our Hex agent.

    The class inherits from AgentBase, which is an abstract class.
    The AgentBase contains the colour property which you can use to get the agent's colour.
    You must implement the make_move method to make the agent functional.
    You CANNOT modify the AgentBase class, otherwise your agent might not function.
    """

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.model = load_model("agents/Group41/weights.pt")
        self.mcts = MCTS(game=None, model=self.model)

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

        # 1. Convert Engine Board to NumPy Board
        board_np = BoardStateNP(board)
        board_np.current_colour = self.colour

        # 2. Handle Swap Rule
        # TODO: Add opening move database
        if turn == 2:
            return Move(-1, -1)

        # 3. Run MCTS
        # TODO: Update time_limit to be dynamic on remaining game time
        best_move = self.mcts.search(board_np, time_limit=0.05)
        return best_move
