# Lecture Notes: Unit III & IV - AI for SPPU SE AI-DS & CSE (AI) 2024

**Course:** Artificial Intelligence  
**Target:** Second Year Engineering Students  
**University:** Savitribai Phule Pune University (SPPU)  
**Programs:** AI-DS & CSE (AI)  
**Academic Year:** 2024

---

## Table of Contents

### Unit III - Adversarial Search and Constraint Satisfaction (9 Hours)
1. [Game Theory Fundamentals](#1-game-theory-fundamentals)
2. [Optimal Decisions in Games](#2-optimal-decisions-in-games)
3. [Minimax Algorithm](#3-minimax-algorithm)
4. [Alpha-Beta Pruning](#4-alpha-beta-pruning)
5. [Monte Carlo Tree Search](#5-monte-carlo-tree-search)
6. [Stochastic Games](#6-stochastic-games)
7. [Partially Observable Games](#7-partially-observable-games)
8. [Limitations of Game Search](#8-limitations-of-game-search)
9. [Constraint Satisfaction Problems](#9-constraint-satisfaction-problems)
10. [Constraint Propagation](#10-constraint-propagation)
11. [Backtracking Search for CSPs](#11-backtracking-search-for-csps)

### Unit IV - Knowledge and Reasoning (9 Hours)
12. [Knowledge-Based Agents](#12-knowledge-based-agents)
13. [The Wumpus World Problem](#13-the-wumpus-world-problem)
14. [Propositional Logic](#14-propositional-logic)
15. [Propositional Theorem Proving](#15-propositional-theorem-proving)
16. [Model Checking](#16-model-checking)
17. [First-Order Logic](#17-first-order-logic)
18. [Inference in First-Order Logic](#18-inference-in-first-order-logic)
19. [Unification](#19-unification)
20. [Forward and Backward Chaining](#20-forward-and-backward-chaining)
21. [Resolution](#21-resolution)

---

# Unit III - Adversarial Search and Constraint Satisfaction

## 1. Game Theory Fundamentals

### What is Game Theory?
Game theory studies strategic decision-making between rational agents. In AI, it helps us understand how intelligent agents should behave when their actions affect each other.

### Key Concepts:
- **Game**: A situation where multiple agents make decisions
- **Player**: An agent making decisions
- **Strategy**: A plan of action for a player
- **Payoff**: The reward/penalty for a particular outcome

### Real-World Example:
Think of a chess match between two players. Each player tries to maximize their chances of winning while minimizing their opponent's chances.

---

## 2. Optimal Decisions in Games

### Zero-Sum Games
In zero-sum games, one player's gain equals another player's loss. The total payoff is always zero.

**Example**: Chess, Checkers, Tic-Tac-Toe

### Game Tree
A tree representation where:
- **Nodes**: Game states
- **Edges**: Possible moves
- **Leaves**: Terminal states (game over)

---

## 3. Minimax Algorithm

### Concept
The Minimax algorithm finds the optimal move by assuming both players play perfectly. The maximizing player tries to maximize the score, while the minimizing player tries to minimize it.

### Pseudocode
```
function MINIMAX(node, depth, maximizing_player):
    if depth == 0 or node is terminal:
        return evaluate(node)
    
    if maximizing_player:
        max_eval = -infinity
        for each child of node:
            eval = MINIMAX(child, depth-1, False)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = +infinity
        for each child of node:
            eval = MINIMAX(child, depth-1, True)
            min_eval = min(min_eval, eval)
        return min_eval
```

### Python Implementation

```python
def minimax(board, depth, is_maximizing_player):
    """
    Minimax algorithm for two-player zero-sum games.
    
    Args:
        board: Current game state
        depth: Maximum search depth
        is_maximizing_player: True if current player is maximizing
    
    Returns:
        Best evaluation score for current player
    """
    # Base case: terminal node or max depth reached
    if depth == 0 or is_game_over(board):
        return evaluate_board(board)
    
    if is_maximizing_player:
        max_evaluation = float('-inf')
        for move in get_possible_moves(board):
            new_board = make_move(board, move)
            evaluation = minimax(new_board, depth - 1, False)
            max_evaluation = max(max_evaluation, evaluation)
        return max_evaluation
    else:
        min_evaluation = float('inf')
        for move in get_possible_moves(board):
            new_board = make_move(board, move)
            evaluation = minimax(new_board, depth - 1, True)
            min_evaluation = min(min_evaluation, evaluation)
        return min_evaluation


def get_best_move(board, depth):
    """
    Find the best move using minimax algorithm.
    
    Args:
        board: Current game state
        depth: Search depth
    
    Returns:
        Best move for current player
    """
    best_move = None
    best_value = float('-inf')
    
    for move in get_possible_moves(board):
        new_board = make_move(board, move)
        move_value = minimax(new_board, depth - 1, False)
        
        if move_value > best_value:
            best_value = move_value
            best_move = move
    
    return best_move


# Helper functions (implementation depends on specific game)
def is_game_over(board):
    """Check if game has ended."""
    pass

def evaluate_board(board):
    """Evaluate board position. Positive favors maximizing player."""
    pass

def get_possible_moves(board):
    """Get all legal moves from current position."""
    pass

def make_move(board, move):
    """Return new board state after making move."""
    pass
```

### Real-World Example: Tic-Tac-Toe
```python
class TicTacToe:
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
    
    def is_winner(self, player):
        """Check if player has won."""
        # Check rows, columns, and diagonals
        for i in range(3):
            if all(self.board[i][j] == player for j in range(3)):
                return True
            if all(self.board[j][i] == player for j in range(3)):
                return True
        
        # Check diagonals
        if all(self.board[i][i] == player for i in range(3)):
            return True
        if all(self.board[i][2-i] == player for i in range(3)):
            return True
        
        return False
    
    def is_board_full(self):
        """Check if board is full."""
        return all(self.board[i][j] != ' ' for i in range(3) for j in range(3))
    
    def evaluate(self):
        """Evaluate current board state."""
        if self.is_winner('X'):
            return 1
        elif self.is_winner('O'):
            return -1
        else:
            return 0
    
    def minimax_tic_tac_toe(self, depth, is_maximizing):
        """Minimax implementation for Tic-Tac-Toe."""
        if self.is_winner('X'):
            return 1
        if self.is_winner('O'):
            return -1
        if self.is_board_full():
            return 0
        
        if is_maximizing:
            best_score = float('-inf')
            for i in range(3):
                for j in range(3):
                    if self.board[i][j] == ' ':
                        self.board[i][j] = 'X'
                        score = self.minimax_tic_tac_toe(depth + 1, False)
                        self.board[i][j] = ' '
                        best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for i in range(3):
                for j in range(3):
                    if self.board[i][j] == ' ':
                        self.board[i][j] = 'O'
                        score = self.minimax_tic_tac_toe(depth + 1, True)
                        self.board[i][j] = ' '
                        best_score = min(score, best_score)
            return best_score
```

---

## 4. Alpha-Beta Pruning

### Concept
Alpha-Beta pruning optimizes the Minimax algorithm by eliminating branches that don't need to be explored, significantly reducing computation time.

### Key Terminology:
- **Alpha (α)**: Best value maximizing player can guarantee
- **Beta (β)**: Best value minimizing player can guarantee
- **Pruning**: Skip evaluating remaining moves when α ≥ β

### Pseudocode
```
function ALPHA_BETA(node, depth, alpha, beta, maximizing_player):
    if depth == 0 or node is terminal:
        return evaluate(node)
    
    if maximizing_player:
        max_eval = -infinity
        for each child of node:
            eval = ALPHA_BETA(child, depth-1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cutoff
        return max_eval
    else:
        min_eval = +infinity
        for each child of node:
            eval = ALPHA_BETA(child, depth-1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_eval
```

### Python Implementation

```python
def alpha_beta_pruning(board, depth, alpha, beta, is_maximizing_player):
    """
    Alpha-Beta pruning algorithm for game tree search.
    
    Args:
        board: Current game state
        depth: Maximum search depth
        alpha: Best value maximizing player can guarantee
        beta: Best value minimizing player can guarantee
        is_maximizing_player: True if current player is maximizing
    
    Returns:
        Best evaluation score with pruning
    """
    # Base case: terminal node or max depth reached
    if depth == 0 or is_game_over(board):
        return evaluate_board(board)
    
    if is_maximizing_player:
        max_evaluation = float('-inf')
        for move in get_possible_moves(board):
            new_board = make_move(board, move)
            evaluation = alpha_beta_pruning(new_board, depth - 1, alpha, beta, False)
            max_evaluation = max(max_evaluation, evaluation)
            alpha = max(alpha, evaluation)
            
            # Beta cutoff: if beta <= alpha, prune remaining branches
            if beta <= alpha:
                break
                
        return max_evaluation
    else:
        min_evaluation = float('inf')
        for move in get_possible_moves(board):
            new_board = make_move(board, move)
            evaluation = alpha_beta_pruning(new_board, depth - 1, alpha, beta, True)
            min_evaluation = min(min_evaluation, evaluation)
            beta = min(beta, evaluation)
            
            # Alpha cutoff: if beta <= alpha, prune remaining branches
            if beta <= alpha:
                break
                
        return min_evaluation


def get_best_move_with_pruning(board, depth):
    """
    Find the best move using alpha-beta pruning.
    
    Args:
        board: Current game state
        depth: Search depth
    
    Returns:
        Best move for current player
    """
    best_move = None
    best_value = float('-inf')
    alpha = float('-inf')
    beta = float('inf')
    
    for move in get_possible_moves(board):
        new_board = make_move(board, move)
        move_value = alpha_beta_pruning(new_board, depth - 1, alpha, beta, False)
        
        if move_value > best_value:
            best_value = move_value
            best_move = move
        
        alpha = max(alpha, move_value)
    
    return best_move


# Example: Alpha-Beta for Chess-like evaluation
class ChessAI:
    def __init__(self):
        self.piece_values = {
            'pawn': 1, 'knight': 3, 'bishop': 3,
            'rook': 5, 'queen': 9, 'king': 100
        }
    
    def evaluate_position(self, board):
        """Simple material evaluation."""
        score = 0
        for row in board:
            for piece in row:
                if piece:
                    value = self.piece_values.get(piece.type, 0)
                    score += value if piece.color == 'white' else -value
        return score
    
    def alpha_beta_chess(self, board, depth, alpha, beta, is_white_turn):
        """Alpha-beta implementation for chess."""
        if depth == 0 or self.is_checkmate(board):
            return self.evaluate_position(board)
        
        if is_white_turn:
            max_eval = float('-inf')
            for move in self.get_legal_moves(board, 'white'):
                new_board = self.make_move(board, move)
                evaluation = self.alpha_beta_chess(new_board, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, evaluation)
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break  # Prune
            return max_eval
        else:
            min_eval = float('inf')
            for move in self.get_legal_moves(board, 'black'):
                new_board = self.make_move(board, move)
                evaluation = self.alpha_beta_chess(new_board, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, evaluation)
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break  # Prune
            return min_eval
```

### Performance Benefits:
- **Time Complexity**: O(b^(d/2)) instead of O(b^d) in best case
- **Space Complexity**: O(d) (same as Minimax)
- **Practical Impact**: Can search twice as deep in same time

---

## 5. Monte Carlo Tree Search

### Concept
Monte Carlo Tree Search (MCTS) uses random sampling to evaluate game positions. Instead of exhaustive search, it runs random simulations to estimate position values.

### Four Phases:
1. **Selection**: Navigate tree using UCB1 formula
2. **Expansion**: Add new child node
3. **Simulation**: Random playout to terminal state
4. **Backpropagation**: Update statistics along path

### UCB1 Formula:
```
UCB1 = average_reward + C * sqrt(ln(parent_visits) / node_visits)
```

### Pseudocode
```
function MCTS(root, iterations):
    for i = 1 to iterations:
        leaf = SELECT(root)
        child = EXPAND(leaf)
        result = SIMULATE(child)
        BACKPROPAGATE(child, result)
    return BEST_CHILD(root)

function SELECT(node):
    while node is not terminal and node is fully expanded:
        node = BEST_UCB1_CHILD(node)
    return node

function EXPAND(node):
    if node is terminal:
        return node
    return CREATE_NEW_CHILD(node)

function SIMULATE(node):
    while node is not terminal:
        node = RANDOM_CHILD(node)
    return EVALUATE(node)

function BACKPROPAGATE(node, result):
    while node is not null:
        node.visits += 1
        node.wins += result
        node = node.parent
```

### Python Implementation

```python
import math
import random
from typing import List, Optional


class MCTSNode:
    """Node in the Monte Carlo Tree Search."""
    
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.wins = 0.0
        self.untried_actions = self.get_legal_actions()
    
    def get_legal_actions(self):
        """Get all legal actions from current state."""
        # Implementation depends on specific game
        pass
    
    def is_terminal(self):
        """Check if this is a terminal state."""
        # Implementation depends on specific game
        pass
    
    def is_fully_expanded(self):
        """Check if all children have been expanded."""
        return len(self.untried_actions) == 0
    
    def ucb1_value(self, c=1.4):
        """Calculate UCB1 value for node selection."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.wins / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def best_child(self, c=1.4):
        """Select best child using UCB1."""
        return max(self.children, key=lambda child: child.ucb1_value(c))
    
    def add_child(self, action, state):
        """Add a new child node."""
        child = MCTSNode(state, parent=self, action=action)
        self.children.append(child)
        return child


class MCTS:
    """Monte Carlo Tree Search implementation."""
    
    def __init__(self, exploration_constant=1.4):
        self.c = exploration_constant
    
    def search(self, root_state, iterations=1000):
        """
        Perform MCTS to find best action.
        
        Args:
            root_state: Initial game state
            iterations: Number of MCTS iterations
            
        Returns:
            Best action found
        """
        root = MCTSNode(root_state)
        
        for _ in range(iterations):
            # Selection
            leaf = self.select(root)
            
            # Expansion
            if not leaf.is_terminal():
                leaf = self.expand(leaf)
            
            # Simulation
            result = self.simulate(leaf)
            
            # Backpropagation
            self.backpropagate(leaf, result)
        
        # Return action of best child
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action
    
    def select(self, node):
        """Selection phase: navigate to leaf using UCB1."""
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child(self.c)
        return node
    
    def expand(self, node):
        """Expansion phase: add new child node."""
        if node.untried_actions:
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            
            # Create new state by applying action
            new_state = self.apply_action(node.state, action)
            child = node.add_child(action, new_state)
            return child
        return node
    
    def simulate(self, node):
        """Simulation phase: random playout to terminal state."""
        current_state = node.state
        
        while not self.is_terminal(current_state):
            actions = self.get_legal_actions(current_state)
            action = random.choice(actions)
            current_state = self.apply_action(current_state, action)
        
        return self.evaluate_terminal(current_state)
    
    def backpropagate(self, node, result):
        """Backpropagation phase: update statistics along path."""
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent
    
    # Game-specific methods (to be implemented for each game)
    def apply_action(self, state, action):
        """Apply action to state and return new state."""
        pass
    
    def get_legal_actions(self, state):
        """Get all legal actions from state."""
        pass
    
    def is_terminal(self, state):
        """Check if state is terminal."""
        pass
    
    def evaluate_terminal(self, state):
        """Evaluate terminal state (1 for win, 0 for loss, 0.5 for draw)."""
        pass
```

### Advantages of MCTS:
- **No domain knowledge required**: Works with just game rules
- **Anytime algorithm**: Can return best move found so far
- **Handles large branching factors**: Better than minimax for complex games
- **Asymmetric tree growth**: Focuses on promising areas

### Real-World Applications:
- **AlphaGo**: Used MCTS + neural networks to defeat world champions
- **Game development**: Used in strategy games like Civilization
- **Planning problems**: Resource allocation, scheduling

---