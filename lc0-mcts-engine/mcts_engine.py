"""
Monte Carlo Tree Search engine for chess using Lc0's pretrained policy.
"""
import numpy as np
from typing import Optional, Tuple
from chess import Board, Move
from mcts_node import MCTSNode
from lc0_policy import Lc0Policy
import time


class MCTSEngine:
    """
    Chess engine using MCTS with Lc0's pretrained neural network policy.
    """
    
    def __init__(self, model_path: str = None, simulations: int = 800, use_gpu: bool = True):
        """
        Initialize the MCTS engine.
        
        Args:
            model_path: Path to Lc0 model file
            simulations: Number of MCTS simulations per move
            use_gpu: Whether to use GPU acceleration
        """
        self.policy = Lc0Policy(model_path, use_gpu)
        self.simulations = simulations
        self.root: Optional[MCTSNode] = None
    
    def search(self, board: Board, time_limit: Optional[float] = None) -> Move:
        """
        Search for the best move using MCTS.
        
        Args:
            board: Current chess board position
            time_limit: Maximum time to spend searching (seconds)
            
        Returns:
            Best move according to MCTS
        """
        # Initialize root node
        self.root = MCTSNode(board)
        
        # Expand root with policy probabilities
        self._expand_node(self.root)
        
        start_time = time.time()
        sim_count = 0
        
        while True:
            # Check time limit
            if time_limit and (time.time() - start_time) >= time_limit:
                break
            
            # Check simulation limit
            if sim_count >= self.simulations:
                break
            
            # MCTS steps
            node = self._select(self.root)
            
            if node.visits > 0 or node.is_terminal():
                # Expand if not terminal
                if not node.is_terminal():
                    self._expand_node(node)
                    node = node.select_child() if node.children else node
                
                # Evaluate using policy network
                value = self._evaluate(node)
            else:
                # First visit - evaluate immediately
                value = self._evaluate(node)
            
            # Backpropagate
            node.backpropagate(value)
            
            sim_count += 1
        
        # Select best move (most visited child)
        if not self.root.children:
            # Fallback: return first legal move
            legal_moves = list(board.legal_moves)
            return legal_moves[0] if legal_moves else None
        
        best_child = max(self.root.children, key=lambda c: c.visits)
        return best_child.move
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Select a node to explore using UCB.
        
        Args:
            node: Starting node
            
        Returns:
            Selected leaf node
        """
        while node.is_expanded and node.children:
            node = node.select_child()
        return node
    
    def _expand_node(self, node: MCTSNode):
        """
        Expand a node by adding children for all legal moves.
        
        Args:
            node: Node to expand
        """
        if node.is_expanded or node.is_terminal():
            return
        
        legal_moves = node.get_legal_moves()
        if not legal_moves:
            return
        
        # Get policy probabilities from neural network
        policy_dict, _ = self.policy.evaluate(node.board)
        
        # Create children for all legal moves
        for move in legal_moves:
            new_board = node.board.copy()
            new_board.push(move)
            child = node.add_child(move, new_board)
            
            # Set prior probability from policy
            child.prior_prob = policy_dict.get(move, 1.0 / len(legal_moves))
        
        node.is_expanded = True
    
    def _evaluate(self, node: MCTSNode) -> float:
        """
        Evaluate a node using the neural network value output.
        
        Args:
            node: Node to evaluate
            
        Returns:
            Value from neural network (-1 to 1, win probability)
        """
        _, value = self.policy.evaluate(node.board)
        
        # If it's black's turn, negate the value
        if not node.board.turn:
            value = -value
        
        # Handle terminal positions
        if node.is_terminal():
            result = node.board.result()
            if result == "1-0":  # White wins
                return 1.0
            elif result == "0-1":  # Black wins
                return -1.0
            else:  # Draw
                return 0.0
        
        return value
    
    def make_move(self, board: Board, time_limit: Optional[float] = None) -> Move:
        """
        Make a move using MCTS search.
        
        Args:
            board: Current board position
            time_limit: Time limit for search (seconds)
            
        Returns:
            Best move found
        """
        return self.search(board, time_limit)
    
    def update_position(self, move: Move):
        """
        Update the engine's internal position after opponent's move.
        This allows the engine to reuse the search tree when possible.
        
        Args:
            move: The move that was played
        """
        if self.root is None:
            return
        
        # Try to find the child node corresponding to the move
        for child in self.root.children:
            if child.move == move:
                self.root = child
                self.root.parent = None  # New root has no parent
                return
        
        # Move not found in tree, reset root
        self.root = None


