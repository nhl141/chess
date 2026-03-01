import chess
import random
import time

def playerMove(board):
    # Recursively request player move until it's legal
    while True:
        try:
            movesan=input("Your move: ")
            move=board.parse_san(movesan)
            break
        except: 
            if movesan == "resign":
                exit()
            if movesan == "Resign":
                exit()
            print("Move is illegal")
            print(board.legal_moves)
    # Move
    board.push(move)

def computerMove(board):
    # Create legal move list
    legalmoves = list(board.legal_moves)
    move = random.choice(legalmoves)
    board.push(move)
    # Computer select random move from list
    # Move
    print("The computer has moved")

print("Welcome to randbot, a chess bot that makes random moves!")
print("You will play as white. The program will end when a position has reached checkmate.")
print(" You can also exit the program by typing 'resign'")
print("You must enter your chess moves in standard algebraic notation")
print("Have fun!")
print("="*150)

board = chess.Board()
print(board)
while True:
    playerMove(board)
    computerMove(board)
    print(board)
    if board.is_checkmate():
        print("Checkmate!")
        exit()

