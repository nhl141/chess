"""
Interactive example of playing against the MCTS engine.
"""
from chess import Board
from chess.engine import SimpleEngine, Limit
from mcts_engine import MCTSEngine


def play_game():
    """Play a game against the MCTS engine."""
    # Initialize engine
    engine = MCTSEngine(
        model_path=None,  # Set to your Lc0 model path
        simulations=400,  # Fewer simulations for faster play
        use_gpu=True
    )
    
    board = Board()
    
    print("=" * 60)
    print("MCTS Chess Engine - Interactive Game")
    print("=" * 60)
    print("\nEnter moves in UCI format (e.g., 'e2e4' or 'e2-e4')")
    print("Type 'quit' to exit, 'resign' to resign, 'print' to see board\n")
    
    while not board.is_game_over():
        print(f"\n{'='*60}")
        print(f"Move {board.fullmove_number}")
        print(f"{'White' if board.turn else 'Black'} to move")
        print(f"{'='*60}")
        print(board)
        print(f"\nFEN: {board.fen()}")
        
        if board.turn:  # Player's turn
            move_str = input("\nYour move: ").strip().lower()
            
            if move_str == 'quit':
                print("Game quit.")
                break
            elif move_str == 'resign':
                print("You resigned!")
                break
            elif move_str == 'print':
                print(board)
                continue
            
            # Parse move
            try:
                move = board.parse_san(move_str) if len(move_str) > 4 else board.parse_uci(move_str.replace('-', ''))
                if move not in board.legal_moves:
                    print("Illegal move! Try again.")
                    continue
                board.push(move)
            except Exception as e:
                print(f"Invalid move: {e}. Try again.")
                continue
        else:  # Engine's turn
            print("\nEngine thinking...")
            move = engine.make_move(board, time_limit=3.0)
            
            if move:
                print(f"Engine plays: {move.uci()}")
                board.push(move)
            else:
                print("Engine has no legal moves!")
                break
    
    # Game over
    if board.is_game_over():
        result = board.result()
        print(f"\n{'='*60}")
        print("Game Over!")
        print(f"Result: {result}")
        if result == "1-0":
            print("White wins!")
        elif result == "0-1":
            print("Black wins!")
        else:
            print("Draw!")
        print(f"{'='*60}")
        print("\nFinal position:")
        print(board)


def engine_vs_engine():
    """Play engine against itself."""
    engine1 = MCTSEngine(model_path=None, simulations=200, use_gpu=True)
    engine2 = MCTSEngine(model_path=None, simulations=200, use_gpu=True)
    
    board = Board()
    
    print("Engine vs Engine")
    print("=" * 60)
    
    move_count = 0
    max_moves = 100
    
    while not board.is_game_over() and move_count < max_moves:
        print(f"\nMove {board.fullmove_number}")
        print(board)
        
        if board.turn:
            print("Engine 1 thinking...")
            move = engine1.make_move(board, time_limit=2.0)
        else:
            print("Engine 2 thinking...")
            move = engine2.make_move(board, time_limit=2.0)
        
        if move:
            print(f"Plays: {move.uci()}")
            board.push(move)
            move_count += 1
        else:
            break
    
    result = board.result()
    print(f"\nResult: {result}")
    print(board)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "self":
        engine_vs_engine()
    else:
        play_game()


