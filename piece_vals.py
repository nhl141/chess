import chess
import math

# Constants
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}


board = chess.Board()

def score(board_obj, cpu_color):
    white = 0
    black = 0
    for square in chess.SQUARES:
        piece = board_obj.piece_at(square)
        if piece:
            piece_value = PIECE_VALUES.get(piece.piece_type,0)
            if piece.color == chess.WHITE:
                white += piece_value
            elif piece.color == chess.BLACK:
                black += piece_value
    if cpu_color == chess.WHITE:
        return white - black
    else:
        return black - white
def minimax(position, depth, alpha, beta, cpu_color):
    is_max_node = (position.turn == cpu_color)
    if depth == 0 or position.is_game_over():
        return score(position, cpu_color)
    if is_max_node:
        maxEval = -math.inf
        for move in position.legal_moves:
            position.push(move)
            eval = minimax(position, depth - 1, alpha, beta, False)
            position.pop()
            maxEval = max(eval, maxEval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return maxEval
    else:
        minEval = math.inf
        for move in position.legal_moves:
            position.push(move)
            eval = minimax(position, depth - 1, alpha, beta, True)
            position.pop()
            minEval = min(eval, minEval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return minEval
def playerMove():
    while True:
        try:
            movesan=input("Your move: ")
            move=board.parse_san(movesan)
            break
        except: 
            if movesan.lower() == "resign":
                exit()
            print("Move is illegal")
            print(board.legal_moves)
    # Move
    board.push(move)
    return False
def cpuMove():
    # using list of legal moves, create a list of all minimax evals per legal move.
    scoretracker = -math.inf
    cpu_move = None
    for move in board.legal_moves:
        # In minimax, input position as the board object after legal move is performed
        # Use minimax to find board with highest score. 
        board.push(move)
        chess.Move.null()
        cpu_move_eval =  minimax(board, 4, -math.inf, math.inf, cpu_color) 
        if cpu_move_eval > scoretracker:
            scoretracker = cpu_move_eval
            cpu_move = move
        board.pop()
    # Move best move
    board.push(cpu_move)
    print(board)
    return True

# Allow player to choose if they are white or black as bool
print(f'Choose colour: W = White, B = Black')
playerColour = input()
# while playerColour != 'W' or playerColor != 'B':
#     print(f'Sorry, that was not one of the options.')
#     print(f'Choose colour: W = White, B = Black')
#     playerColour = input()

if playerColour in ['W','White','w','white']:
    playerTurn=True
    cpu_color = chess.BLACK
else:
    playerTurn=False
    cpu_color = chess.WHITE

print(board)
# while the position is not in checkmate: 
while not board.is_game_over():
    if playerTurn:
        playerTurn = playerMove()
    else:
        playerTurn = cpuMove()