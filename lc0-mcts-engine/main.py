"""
Example usage of the MCTS engine with Lc0 policy.
"""
from chess import Board
from mcts_engine import MCTSEngine


def main():
    """Example of using the engine to play a game."""
    # Initialize engine
    # You can provide a path to an Lc0 model file here
    # Download models from: https://lczero.org/networks/
    engine = MCTSEngine(
        model_path=None,  # Set to your model path, e.g., "path/to/model.pb.gz"
        simulations=800,
        use_gpu=True
    )
    
    # Create a board
    board = Board()
    
    print("MCTS Engine with Lc0 Policy")
    print("=" * 50)
    print("\nInitial position:")
    print(board)
    print()
    
    # Make a move
    print("Thinking...")
    move = engine.make_move(board, time_limit=5.0)
    
    if move:
        print(f"Engine plays: {move}")
        board.push(move)
        print("\nAfter move:")
        print(board)
        print(f"\nFEN: {board.fen()}")
    else:
        print("No legal moves available!")


if __name__ == "__main__":
    main()


