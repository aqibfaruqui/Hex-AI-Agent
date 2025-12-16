from agents.Group41.mcts import MCTS
from agents.Group41.board_state import BoardStateNP
from src.Move import Move
from src.Board import Board
import random
import matplotlib.pyplot as plt

def run_profile():
    print("Transposition Table Profiling")

    x_stones = []
    y_speedup = []
    y_hits = []

    for stones in range(0, 115, 10):

        # We play some dummy moves to create a complex state
        board = BoardStateNP(Board(11))
        moves = [(x, y) for x in range(11) for y in range(11)]
        random.seed(67)
        random.shuffle(moves)
        
        # Place n stones (Branching factor ~(121-stones))
        for i in range(stones):
            x, y = moves[i]
            board.play_move(Move(x, y))

        # TEST 1: Without Transposition Table
        mcts_old = MCTS(game=None, model=None, tt=False)
        mcts_old.search_for_selfplay(board, time_limit=5.0)
        sims_old = mcts_old.total_simulations
        hits_old = mcts_old.transposition_hits     # Should be 0

        # TEST 2: With Transposition Table
        mcts_new = MCTS(game=None, model=None, tt=True)
        mcts_new.search_for_selfplay(board, time_limit=5.0)
        sims_new = mcts_new.total_simulations
        hits_new = mcts_new.transposition_hits

        # Analysis
        diff = sims_new - sims_old
        speedup_percent = (diff / sims_old) * 100 if sims_old > 0 else 0
        hit_rate = (hits_new / sims_new * 100) if sims_new > 0 else 0
        
        x_stones.append(stones)
        y_speedup.append(speedup_percent)
        y_hits.append(hit_rate)
        
        print(f"Stones: {stones:<3} | Sims(Old): {sims_old:<5} | Sims(New): {sims_new:<5} | Speedup: {speedup_percent:+.1f}%")


    print("\nGenerating Graph...")    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary Axis (Speedup - Red)
    color = 'tab:red'
    ax1.set_xlabel('Game Stage (Stones Placed)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Increase in Simulations (%)', color=color, fontsize=12, fontweight='bold')
    ax1.plot(x_stones, y_speedup, color=color, marker='o', linewidth=2, label='Simulation Speedup')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Secondary Axis (Cache Hits - Blue)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Transposition Hit Rate (%)', color=color, fontsize=12, fontweight='bold')
    ax2.plot(x_stones, y_hits, color=color, linestyle='--', marker='x', label='Cache Hit Rate')
    ax2.tick_params(axis='y', labelcolor=color)

    # Context Labels
    plt.title('Impact of Transposition Tables on Search Efficiency', fontsize=14)
    plt.text(20, max(y_hits)*0.8, 'Opening\n(Sparse)', color='gray', ha='center')
    plt.text(80, max(y_hits)*0.8, 'Endgame\n(Dense)', color='gray', ha='center')

    # Save to file
    output_filename = 'agents/Group41/experiments/tt_performance.png'
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"Graph saved successfully to {output_filename}")

if __name__ == "__main__":
    run_profile()