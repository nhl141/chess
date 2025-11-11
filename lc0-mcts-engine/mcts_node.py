"""
MCTS Node implementation for chess engine.
Represents a node in the Monte Carlo Tree Search tree.
"""
from typing import Optional, List, Dict
import numpy as np
from chess import Move, Board


class MCTSNode:
    """Node in the MCTS search tree."""
    
    def __init__(self, board: Board, move: Optional[Move] = None, parent: Optional['MCTSNode'] = None):
        """
        Initialize a MCTS node.
        
        Args:
            board: The chess board state at this node
            move: The move that led to this node (None for root)
            parent: Parent node in the tree
        """
        self.board = board.copy()
        self.move = move
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.value_sum = 0.0  # Sum of all values from simulations
        self.prior_prob = 0.0  # Prior probability from policy network
        self.is_expanded = False
        
    @property
    def value(self) -> float:
        """Average value of this node."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    @property
    def ucb_score(self) -> float:
        """Calculate UCB (Upper Confidence Bound) score for selection."""
        if self.visits == 0:
            return float('inf')
        
        # UCB formula: value + c * sqrt(log(parent_visits) / visits) * prior
        if self.parent is None:
            return self.value
        
        exploration_constant = 1.5  # c in UCB formula
        parent_visits = self.parent.visits if self.parent else 1
        
        exploitation = self.value
        exploration = exploration_constant * np.sqrt(np.log(parent_visits) / self.visits)
        prior_boost = self.prior_prob * np.sqrt(parent_visits) / (1 + self.visits)
        
        return exploitation + exploration + prior_boost
    
    def add_child(self, move: Move, board: Board) -> 'MCTSNode':
        """Add a child node to this node."""
        child = MCTSNode(board, move, self)
        self.children.append(child)
        return child
    
    def is_terminal(self) -> bool:
        """Check if this node is a terminal state (game over)."""
        return self.board.is_game_over()
    
    def get_legal_moves(self) -> List[Move]:
        """Get list of legal moves from this position."""
        return list(self.board.legal_moves)
    
    def select_child(self) -> Optional['MCTSNode']:
        """Select the best child node using UCB scores."""
        if not self.children:
            return None
        
        # Select child with highest UCB score
        scores = [child.ucb_score for child in self.children]
        best_idx = np.argmax(scores)
        return self.children[best_idx]
    
    def backpropagate(self, value: float):
        """Backpropagate the value up the tree."""
        self.visits += 1
        self.value_sum += value
        
        # From opponent's perspective, value is negated
        if self.parent is not None:
            self.parent.backpropagate(-value)


