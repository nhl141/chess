# Lc0 MCTS Chess Engine

A chess engine that fully utilizes Lc0's pretrained neural network policy with Monte Carlo Tree Search (MCTS) algorithm.

## Features

- **MCTS Algorithm**: Implements Monte Carlo Tree Search with UCB (Upper Confidence Bound) selection
- **Lc0 Policy Integration**: Uses Lc0's pretrained neural network for move probabilities and position evaluation
- **Efficient Search**: Tree reuse and policy-guided exploration
- **GPU Support**: Optional GPU acceleration for neural network inference

## Architecture

### Components

1. **MCTSNode** (`mcts_node.py`): Represents nodes in the search tree with UCB scoring
2. **Lc0Policy** (`lc0_policy.py`): Wrapper for loading and using Lc0 neural network models
3. **MCTSEngine** (`mcts_engine.py`): Main engine that combines MCTS with policy network

### MCTS Algorithm

The engine implements the standard MCTS algorithm with four phases:

1. **Selection**: Traverse the tree using UCB scores to find a leaf node
2. **Expansion**: Add child nodes for all legal moves, weighted by policy probabilities
3. **Simulation/Evaluation**: Use the neural network to evaluate the position
4. **Backpropagation**: Update node values and visit counts up the tree

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Download an Lc0 model (optional but recommended):
   - Visit [lczero.org/networks](https://lczero.org/networks/) to download pretrained models
   - Models are typically in `.pb.gz` or `.onnx` format
   - Place the model file in your project directory

## Usage

### Basic Usage

```python
from chess import Board
from mcts_engine import MCTSEngine

# Initialize engine
engine = MCTSEngine(
    model_path="path/to/your/model.pb.gz",  # Optional: path to Lc0 model
    simulations=800,  # Number of MCTS simulations per move
    use_gpu=True  # Enable GPU acceleration if available
)

# Create a board
board = Board()

# Get engine's move
move = engine.make_move(board, time_limit=5.0)
board.push(move)
```

### Running the Example

```bash
python main.py
```

## Configuration

### Engine Parameters

- `model_path`: Path to Lc0 neural network model file (if None, uses uniform random policy)
- `simulations`: Number of MCTS simulations per move (default: 800)
- `use_gpu`: Whether to use GPU for neural network inference (default: True)

### Search Parameters

- `time_limit`: Maximum time to spend searching (in seconds)
- The engine will use either the time limit or simulation count, whichever is reached first

## Model Format Support

The engine supports multiple model formats:

- **ONNX** (`.onnx`): Primary format, uses ONNX Runtime
- **TensorFlow** (`.pb`, `.pb.gz`): Uses TensorFlow backend
- **Fallback**: If no model is provided, uses uniform random policy

## How It Works

1. **Policy Network**: The Lc0 neural network provides:
   - **Policy**: Probability distribution over all legal moves
   - **Value**: Expected win probability from the current position

2. **MCTS Search**: 
   - Uses policy probabilities as priors for expansion
   - Uses value estimates for evaluation
   - Balances exploration vs exploitation using UCB formula

3. **Move Selection**: 
   - After simulations, selects the most visited child node
   - This represents the move explored most often

## Performance Tips

- **GPU Acceleration**: Enable GPU support for faster neural network inference
- **Simulation Count**: More simulations = stronger play but slower
- **Time Limits**: Use time limits for consistent move timing
- **Model Quality**: Stronger Lc0 models will produce better play

## Limitations

- The board encoding is simplified (full Lc0 uses 119 input planes)
- Move encoding is simplified (full Lc0 uses specific move representation)
- For production use, consider using the full Lc0 implementation with proper encoding

## License

This is an educational implementation. For production chess engines, consider using the official Lc0 implementation.

## References

- [Lc0 Official Website](https://lczero.org/)
- [MCTS Algorithm](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)


