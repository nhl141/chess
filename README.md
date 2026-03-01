# How to use 
- Run randbot.py to play against a random chess engine.
- Run piece_val.py to play against a minimax algorithm that evaluates based on piece values

## randbot.py
Fetches a list of all legal moves in a given game state, and randomly selects a move.

## piece_vals.py
The chess engine uses the minimax algorithm to deterministically evaluate future board positions, using the piece value mapping: Q=9, R=5, B=3, N=3, P=1. 
The engine implements alpha-beta pruning to optimize processing speed, able to process up to four moves ahead under 10s.

However, evaluating board positions by piece-value is quite basic, and low depth minimax searches fail to accurately depict good board positions. Increasing tree depth by even on move scales computation time exponentially. More sophisticated evaluation functions would increase the quality of each depth evaluation level. Furthermore, the engine frequently produced random and poor opening positions, leading it to consistently lose. Hardcoding the first few moves may produce more reasonable openings.
