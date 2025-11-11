"""
Wrapper for Lc0's pretrained neural network policy.
Loads and uses the neural network to evaluate chess positions.
"""
import numpy as np
from typing import Tuple, Dict, List
from chess import Board, Move
import os


class Lc0Policy:
    """
    Wrapper for Lc0's pretrained neural network.
    Provides policy (move probabilities) and value (position evaluation) predictions.
    """
    
    def __init__(self, model_path: str = None, use_gpu: bool = True):
        """
        Initialize the Lc0 policy network.
        
        Args:
            model_path: Path to the Lc0 model file (.pb, .onnx, or .pb.gz)
            use_gpu: Whether to use GPU acceleration
        """
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the neural network model."""
        # Try to load model using different backends
        if self.model_path and os.path.exists(self.model_path):
            try:
                # Try ONNX Runtime first (most common)
                import onnxruntime as ort
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
                self.model = ort.InferenceSession(self.model_path, providers=providers)
                self.backend = 'onnx'
                print(f"Loaded Lc0 model from {self.model_path} using ONNX Runtime")
                return
            except ImportError:
                print("ONNX Runtime not available, trying TensorFlow...")
            except Exception as e:
                print(f"Failed to load with ONNX: {e}, trying TensorFlow...")
            
            try:
                # Fallback to TensorFlow
                import tensorflow as tf
                self.model = tf.saved_model.load(self.model_path)
                self.backend = 'tensorflow'
                print(f"Loaded Lc0 model from {self.model_path} using TensorFlow")
                return
            except ImportError:
                print("TensorFlow not available, using dummy policy...")
            except Exception as e:
                print(f"Failed to load with TensorFlow: {e}, using dummy policy...")
        
        # Fallback to dummy policy if no model loaded
        self.backend = 'dummy'
        print("Warning: Using dummy policy (uniform random). Please provide a valid Lc0 model.")
    
    def board_to_input(self, board: Board) -> np.ndarray:
        """
        Convert a chess board to the input format expected by the neural network.
        Lc0 uses a 8x8x119 input representation (planes x height x width).
        
        Args:
            board: The chess board position
            
        Returns:
            Input tensor for the neural network
        """
        # Simplified representation - in practice, Lc0 uses 119 input planes
        # This is a basic implementation; full Lc0 encoding is more complex
        planes = []
        
        # Piece positions (6 piece types * 2 colors = 12 planes)
        for color in [True, False]:  # White, Black
            for piece_type in [1, 2, 3, 4, 5, 6]:  # Pawn, Knight, Bishop, Rook, Queen, King
                plane = np.zeros((8, 8), dtype=np.float32)
                for square in board.pieces(piece_type, color):
                    row, col = divmod(square, 8)
                    plane[7 - row, col] = 1.0  # Flip row for board orientation
                planes.append(plane)
        
        # Additional planes for castling rights, en passant, etc.
        # (Simplified - full implementation would have 119 planes)
        for _ in range(107):  # Placeholder for other planes
            planes.append(np.zeros((8, 8), dtype=np.float32))
        
        # Stack planes: (119, 8, 8) -> reshape or transpose as needed
        input_tensor = np.stack(planes, axis=0).astype(np.float32)
        
        # Add batch dimension: (1, 119, 8, 8)
        return np.expand_dims(input_tensor, axis=0)
    
    def moves_to_output_indices(self, board: Board, moves: List[Move]) -> Dict[Move, int]:
        """
        Map legal moves to output indices in the policy vector.
        
        Args:
            board: The chess board
            moves: List of legal moves
            
        Returns:
            Dictionary mapping moves to output indices
        """
        # Simplified mapping - in practice, Lc0 uses a specific move encoding
        # This maps each move to an index in the 1858-dimensional policy output
        move_to_idx = {}
        for idx, move in enumerate(moves):
            # In real Lc0, moves are encoded as (from_square, to_square, promotion)
            # This is a simplified version
            move_to_idx[move] = idx % 1858  # Lc0 has 1858 possible move outputs
        return move_to_idx
    
    def evaluate(self, board: Board) -> Tuple[Dict[Move, float], float]:
        """
        Evaluate a chess position using the neural network.
        
        Args:
            board: The chess board position
            
        Returns:
            Tuple of (policy_dict, value) where:
            - policy_dict: Dictionary mapping moves to probabilities
            - value: Expected win probability from current player's perspective
        """
        if self.backend == 'dummy':
            # Improved heuristic-based policy (when no model is loaded)
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return {}, 0.0
            
            # Score moves using basic chess heuristics
            move_scores = {}
            for move in legal_moves:
                score = 0.0
                
                # Prefer captures (especially higher value pieces)
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    piece_values = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}  # Pawn, Knight, Bishop, Rook, Queen, King
                    score += piece_values.get(captured_piece.piece_type, 1) * 10
                
                # Test the move
                board.push(move)
                
                # Prefer moves that give check
                if board.is_check():
                    score += 5
                
                # Prefer moves that get out of check
                if board.was_into_check():
                    score += 3
                
                board.pop()
                
                # Prefer center control
                center_squares = [27, 28, 35, 36]  # d4, d5, e4, e5
                if move.to_square in center_squares:
                    score += 2
                
                # Prefer castling
                if board.is_castling(move):
                    score += 3
                
                # Prefer development (moving pieces from back rank)
                if board.piece_at(move.from_square):
                    if board.turn:  # White
                        if move.from_square // 8 == 0:  # From back rank
                            score += 1
                    else:  # Black
                        if move.from_square // 8 == 7:  # From back rank
                            score += 1
                
                board.pop()
                
                # Base score to avoid zeros
                move_scores[move] = max(score, 0.1)
            
            # Convert scores to probabilities (softmax-like)
            total_score = sum(move_scores.values())
            policy = {move: score / total_score for move, score in move_scores.items()}
            
            # Basic position evaluation (very simple)
            value = 0.0
            if board.is_checkmate():
                value = -1.0 if board.turn else 1.0
            elif board.is_stalemate() or board.is_insufficient_material():
                value = 0.0
            else:
                # Simple material count
                material = sum(len(board.pieces(pt, True)) - len(board.pieces(pt, False)) 
                              for pt in [1, 2, 3, 4, 5])
                value = np.tanh(material / 20.0)  # Normalize to [-1, 1]
                if not board.turn:
                    value = -value
            
            return policy, value
        
        try:
            # Prepare input
            input_tensor = self.board_to_input(board)
            
            if self.backend == 'onnx':
                # Get input and output names
                input_name = self.model.get_inputs()[0].name
                output_names = [output.name for output in self.model.get_outputs()]
                
                # Run inference
                outputs = self.model.run(output_names, {input_name: input_tensor})
                
                # Lc0 typically outputs: policy (1858), value (1)
                policy_logits = outputs[0][0]  # Shape: (1858,)
                value_output = outputs[1][0][0] if len(outputs) > 1 else 0.0  # Scalar value
                
            elif self.backend == 'tensorflow':
                # TensorFlow inference
                policy_logits, value_output = self.model(input_tensor)
                policy_logits = policy_logits.numpy()[0]
                value_output = float(value_output.numpy()[0])
            
            # Convert policy logits to probabilities
            policy_probs = self._softmax(policy_logits)
            
            # Map to legal moves
            legal_moves = list(board.legal_moves)
            move_to_idx = self.moves_to_output_indices(board, legal_moves)
            
            policy_dict = {}
            total_prob = 0.0
            for move in legal_moves:
                idx = move_to_idx[move]
                prob = policy_probs[idx]
                policy_dict[move] = prob
                total_prob += prob
            
            # Normalize to ensure probabilities sum to 1
            if total_prob > 0:
                policy_dict = {move: prob / total_prob for move, prob in policy_dict.items()}
            
            # Value is from -1 to 1, representing win probability
            # Convert to 0-1 range where 0.5 is draw
            value = float(value_output)
            
            return policy_dict, value
            
        except Exception as e:
            print(f"Error evaluating position: {e}")
            # Fallback to dummy policy
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return {}, 0.0
            policy = {move: 1.0 / len(legal_moves) for move in legal_moves}
            return policy, 0.0
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        return exp_x / np.sum(exp_x)

