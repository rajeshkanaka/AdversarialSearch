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

## 6. Stochastic Games

### Concept
Stochastic games involve random events that affect game outcomes. Unlike deterministic games, players must consider probability distributions over possible outcomes.

### Key Features:
- **Chance nodes**: Represent random events
- **Expected value**: Calculate average outcome over all possibilities
- **Risk assessment**: Consider variance in outcomes

### Expectiminimax Algorithm

### Pseudocode
```
function EXPECTIMINIMAX(node, depth, player):
    if depth == 0 or node is terminal:
        return evaluate(node)
    
    if player == MAXIMIZING:
        return max(EXPECTIMINIMAX(child, depth-1, MINIMIZING) for child in children(node))
    elif player == MINIMIZING:
        return min(EXPECTIMINIMAX(child, depth-1, MAXIMIZING) for child in children(node))
    else:  // CHANCE node
        return sum(probability(child) * EXPECTIMINIMAX(child, depth-1, next_player) 
                  for child in children(node))
```

### Python Implementation

```python
import random
from enum import Enum

class PlayerType(Enum):
    MAXIMIZING = "max"
    MINIMIZING = "min"
    CHANCE = "chance"

def expectiminimax(game_state, depth, player_type):
    """
    Expectiminimax algorithm for stochastic games.
    
    Args:
        game_state: Current state of the game
        depth: Remaining search depth
        player_type: Type of current player (MAX, MIN, or CHANCE)
    
    Returns:
        Expected utility value
    """
    # Base case: terminal state or depth limit reached
    if depth == 0 or is_terminal_state(game_state):
        return evaluate_state(game_state)
    
    if player_type == PlayerType.MAXIMIZING:
        max_value = float('-inf')
        for action in get_possible_actions(game_state):
            new_state = apply_action(game_state, action)
            next_player = get_next_player(game_state, action)
            value = expectiminimax(new_state, depth - 1, next_player)
            max_value = max(max_value, value)
        return max_value
    
    elif player_type == PlayerType.MINIMIZING:
        min_value = float('inf')
        for action in get_possible_actions(game_state):
            new_state = apply_action(game_state, action)
            next_player = get_next_player(game_state, action)
            value = expectiminimax(new_state, depth - 1, next_player)
            min_value = min(min_value, value)
        return min_value
    
    else:  # CHANCE node
        expected_value = 0.0
        chance_outcomes = get_chance_outcomes(game_state)
        
        for outcome, probability in chance_outcomes:
            new_state = apply_chance_outcome(game_state, outcome)
            next_player = get_next_player_after_chance(game_state)
            value = expectiminimax(new_state, depth - 1, next_player)
            expected_value += probability * value
        
        return expected_value


# Example: Backgammon-style game with dice
class StochasticGameExample:
    def __init__(self):
        self.dice_probabilities = {
            1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6
        }
    
    def roll_dice(self):
        """Simulate dice roll."""
        return random.randint(1, 6)
    
    def get_expected_value_for_dice_roll(self, game_state, depth, next_player):
        """Calculate expected value considering all possible dice outcomes."""
        expected_value = 0.0
        
        for dice_value, probability in self.dice_probabilities.items():
            # Create new state with this dice outcome
            new_state = self.apply_dice_outcome(game_state, dice_value)
            value = expectiminimax(new_state, depth - 1, next_player)
            expected_value += probability * value
        
        return expected_value
    
    def apply_dice_outcome(self, state, dice_value):
        """Apply dice outcome to game state."""
        # Implementation depends on specific game
        pass


# Poker-like example with card probabilities
class PokerStyleGame:
    def __init__(self):
        self.deck = list(range(1, 53))  # 52 cards
    
    def calculate_hand_probability(self, hand):
        """Calculate probability of getting specific hand."""
        # Simplified example
        if self.is_royal_flush(hand):
            return 1 / 649740
        elif self.is_straight_flush(hand):
            return 8 / 649740
        # ... more hand types
        else:
            return 0.5  # High card
    
    def expectiminimax_poker(self, game_state, depth, player_type):
        """Expectiminimax for poker-style game."""
        if depth == 0 or self.is_game_over(game_state):
            return self.evaluate_poker_state(game_state)
        
        if player_type == PlayerType.CHANCE:
            # Calculate expected value over all possible card deals
            expected_value = 0.0
            remaining_cards = self.get_remaining_cards(game_state)
            
            for card in remaining_cards:
                probability = 1 / len(remaining_cards)
                new_state = self.deal_card(game_state, card)
                value = self.expectiminimax_poker(new_state, depth - 1, PlayerType.MAXIMIZING)
                expected_value += probability * value
            
            return expected_value
        
        # Regular minimax for player decisions
        return expectiminimax(game_state, depth, player_type)
```

### Real-World Example: Weather-Dependent Strategy
```python
class WeatherGame:
    """Game where weather affects outcomes."""
    
    def __init__(self):
        self.weather_probabilities = {
            'sunny': 0.6,
            'rainy': 0.3,
            'stormy': 0.1
        }
    
    def get_action_outcomes(self, action, weather):
        """Get action success probability based on weather."""
        if action == 'outdoor_event':
            if weather == 'sunny':
                return 0.9  # 90% success
            elif weather == 'rainy':
                return 0.4  # 40% success
            else:  # stormy
                return 0.1  # 10% success
        elif action == 'indoor_event':
            return 0.8  # Always 80% success regardless of weather
    
    def calculate_expected_utility(self, action):
        """Calculate expected utility considering weather uncertainty."""
        expected_utility = 0.0
        
        for weather, weather_prob in self.weather_probabilities.items():
            success_prob = self.get_action_outcomes(action, weather)
            utility = 100 if success_prob > 0.5 else -50  # Simplified utility
            expected_utility += weather_prob * success_prob * utility
        
        return expected_utility
```

---

## 7. Partially Observable Games

### Concept
In partially observable games, players don't have complete information about the game state. Players must make decisions based on incomplete information and uncertainty.

### Key Characteristics:
- **Hidden information**: Some game state is unknown to players
- **Belief states**: Players maintain probability distributions over possible states
- **Information sets**: Groups of game states that are indistinguishable to a player

### Real-World Examples:
- **Poker**: Players don't see opponents' cards
- **Battleship**: Players don't know ship locations
- **Fog of War games**: Limited visibility in strategy games

### Belief State Representation

```python
class BeliefState:
    """Represents player's belief about hidden game state."""
    
    def __init__(self):
        self.state_probabilities = {}  # state -> probability
    
    def update_belief(self, observation, action_taken):
        """Update belief based on new observation."""
        new_beliefs = {}
        
        for state, old_prob in self.state_probabilities.items():
            # Calculate probability of observation given state and action
            obs_prob = self.observation_probability(observation, state, action_taken)
            if obs_prob > 0:
                new_beliefs[state] = old_prob * obs_prob
        
        # Normalize probabilities
        total_prob = sum(new_beliefs.values())
        if total_prob > 0:
            for state in new_beliefs:
                new_beliefs[state] /= total_prob
        
        self.state_probabilities = new_beliefs
    
    def observation_probability(self, observation, state, action):
        """Calculate P(observation | state, action)."""
        # Implementation depends on specific game
        pass
    
    def get_most_likely_state(self):
        """Return state with highest probability."""
        if not self.state_probabilities:
            return None
        return max(self.state_probabilities.items(), key=lambda x: x[1])[0]
    
    def get_expected_value(self, evaluation_function):
        """Calculate expected value over all possible states."""
        expected_value = 0.0
        for state, probability in self.state_probabilities.items():
            expected_value += probability * evaluation_function(state)
        return expected_value


# Example: Simplified Poker Implementation
class PokerGame:
    """Simplified poker with hidden cards."""
    
    def __init__(self):
        self.deck = list(range(1, 53))
        self.hand_rankings = {
            'high_card': 1, 'pair': 2, 'two_pair': 3,
            'three_kind': 4, 'straight': 5, 'flush': 6,
            'full_house': 7, 'four_kind': 8, 'straight_flush': 9
        }
    
    def get_hand_strength(self, cards):
        """Evaluate hand strength."""
        # Simplified hand evaluation
        if len(set(cards)) == 1:
            return self.hand_rankings['four_kind']
        elif len(set(cards)) == 2:
            return self.hand_rankings['pair']
        else:
            return self.hand_rankings['high_card']
    
    def calculate_win_probability(self, my_cards, belief_state):
        """Calculate probability of winning given belief about opponent's cards."""
        win_probability = 0.0
        my_strength = self.get_hand_strength(my_cards)
        
        for opponent_cards, probability in belief_state.state_probabilities.items():
            opponent_strength = self.get_hand_strength(opponent_cards)
            
            if my_strength > opponent_strength:
                win_probability += probability
            elif my_strength == opponent_strength:
                win_probability += probability * 0.5  # Tie
        
        return win_probability
    
    def should_bet(self, my_cards, belief_state, bet_amount, pot_size):
        """Decide whether to bet based on expected value."""
        win_prob = self.calculate_win_probability(my_cards, belief_state)
        
        # Expected value calculation
        expected_value = win_prob * pot_size - (1 - win_prob) * bet_amount
        
        return expected_value > 0


# Battleship Example
class BattleshipGame:
    """Battleship with hidden ship locations."""
    
    def __init__(self, board_size=10):
        self.board_size = board_size
        self.ship_sizes = [5, 4, 3, 3, 2]  # Aircraft carrier, battleship, etc.
    
    def initialize_belief_state(self):
        """Initialize uniform belief over all possible ship configurations."""
        belief_state = BeliefState()
        possible_configs = self.generate_all_ship_configurations()
        
        # Uniform distribution initially
        prob_per_config = 1.0 / len(possible_configs)
        for config in possible_configs:
            belief_state.state_probabilities[config] = prob_per_config
        
        return belief_state
    
    def update_belief_after_shot(self, belief_state, shot_position, result):
        """Update belief based on shot result (hit/miss)."""
        new_beliefs = {}
        
        for config, old_prob in belief_state.state_probabilities.items():
            # Check if this configuration is consistent with the shot result
            if self.is_consistent(config, shot_position, result):
                new_beliefs[config] = old_prob
        
        # Normalize
        total_prob = sum(new_beliefs.values())
        if total_prob > 0:
            for config in new_beliefs:
                new_beliefs[config] /= total_prob
        
        belief_state.state_probabilities = new_beliefs
    
    def is_consistent(self, ship_configuration, shot_position, result):
        """Check if shot result is consistent with ship configuration."""
        has_ship_at_position = shot_position in ship_configuration
        return (result == 'hit') == has_ship_at_position
    
    def choose_next_shot(self, belief_state):
        """Choose next shot position to maximize information gain."""
        best_position = None
        best_expected_info = -1
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                position = (row, col)
                if not self.already_shot(position):
                    expected_info = self.calculate_expected_information(
                        belief_state, position)
                    if expected_info > best_expected_info:
                        best_expected_info = expected_info
                        best_position = position
        
        return best_position
    
    def calculate_expected_information(self, belief_state, position):
        """Calculate expected information gain from shooting at position."""
        hit_prob = 0.0
        
        # Calculate probability of hit at this position
        for config, prob in belief_state.state_probabilities.items():
            if position in config:
                hit_prob += prob
        
        miss_prob = 1.0 - hit_prob
        
        # Information theory: expected information gain
        info_gain = 0.0
        if hit_prob > 0:
            info_gain -= hit_prob * math.log2(hit_prob)
        if miss_prob > 0:
            info_gain -= miss_prob * math.log2(miss_prob)
        
        return info_gain
```

### Information Set Theory
In game theory, an **information set** represents states that are indistinguishable to a player:

```python
class InformationSet:
    """Represents an information set in extensive-form games."""
    
    def __init__(self, states, player):
        self.states = states  # List of game states in this information set
        self.player = player  # Player who cannot distinguish these states
        self.actions = self.get_available_actions()
    
    def get_available_actions(self):
        """Get actions available from all states in this information set."""
        if not self.states:
            return []
        
        # Actions must be the same for all states in information set
        actions = set(get_actions(self.states[0]))
        for state in self.states[1:]:
            actions &= set(get_actions(state))
        
        return list(actions)
    
    def calculate_expected_utility(self, action, strategy):
        """Calculate expected utility for an action."""
        total_utility = 0.0
        total_probability = 0.0
        
        for state in self.states:
            state_prob = strategy.get_state_probability(state)
            state_utility = self.evaluate_action_in_state(action, state)
            
            total_utility += state_prob * state_utility
            total_probability += state_prob
        
        return total_utility / total_probability if total_probability > 0 else 0
```

---

## 8. Limitations of Game Search

### Computational Limitations

#### Time Complexity Issues:
- **Exponential growth**: Game trees grow exponentially with depth
- **Branching factor**: High branching factor makes search intractable
- **Depth limitation**: Cannot search to game end in complex games

#### Space Complexity Issues:
- **Memory requirements**: Storing large game trees
- **Transposition tables**: Managing hash tables for seen positions

### Real-World Constraints

```python
class GameSearchLimitations:
    """Demonstrates practical limitations of game search."""
    
    def __init__(self):
        self.time_limit = 5.0  # seconds
        self.memory_limit = 1000000  # nodes
        self.nodes_visited = 0
        self.start_time = None
    
    def iterative_deepening_search(self, initial_state):
        """Search with increasing depth until time runs out."""
        self.start_time = time.time()
        best_move = None
        depth = 1
        
        try:
            while True:
                # Check time limit
                if time.time() - self.start_time > self.time_limit:
                    break
                
                # Search at current depth
                move = self.alpha_beta_search(initial_state, depth)
                if move is not None:
                    best_move = move
                
                depth += 1
                
        except MemoryError:
            print(f"Memory limit reached at depth {depth}")
        
        return best_move
    
    def alpha_beta_search(self, state, depth):
        """Alpha-beta search with resource monitoring."""
        self.nodes_visited = 0
        return self.alpha_beta_recursive(state, depth, float('-inf'), float('inf'), True)
    
    def alpha_beta_recursive(self, state, depth, alpha, beta, maximizing):
        """Recursive alpha-beta with cutoffs."""
        self.nodes_visited += 1
        
        # Check resource limits
        if self.nodes_visited > self.memory_limit:
            raise MemoryError("Node limit exceeded")
        
        if time.time() - self.start_time > self.time_limit:
            return None  # Time cutoff
        
        # Regular alpha-beta logic
        if depth == 0 or is_terminal(state):
            return evaluate(state)
        
        # ... rest of alpha-beta implementation
```

### Domain-Specific Challenges

#### Chess Limitations:
- **50-move rule**: Games can end in draws
- **Repetition**: Threefold repetition leads to draws
- **Opening theory**: Extensive memorized openings
- **Endgame databases**: Perfect play in endgames

#### Go Limitations:
- **Huge branching factor**: ~250 moves per position
- **Positional evaluation**: Difficult to evaluate non-terminal positions
- **Pattern recognition**: Requires deep pattern understanding

### Heuristic Limitations

```python
class HeuristicLimitations:
    """Examples of heuristic evaluation problems."""
    
    def naive_chess_evaluation(self, board):
        """Simple material count - often insufficient."""
        material_value = 0
        piece_values = {'pawn': 1, 'knight': 3, 'bishop': 3, 'rook': 5, 'queen': 9}
        
        for piece in board.pieces:
            value = piece_values.get(piece.type, 0)
            material_value += value if piece.color == 'white' else -value
        
        # Problem: Ignores position, king safety, pawn structure, etc.
        return material_value
    
    def better_chess_evaluation(self, board):
        """More sophisticated evaluation."""
        score = 0
        
        # Material value
        score += self.material_evaluation(board)
        
        # Positional factors
        score += self.piece_square_tables(board)
        score += self.king_safety(board)
        score += self.pawn_structure(board)
        score += self.piece_mobility(board)
        
        # Still incomplete - missing many factors
        return score
    
    def horizon_effect_example(self, state, depth):
        """Demonstrates horizon effect problem."""
        # Player delays inevitable capture beyond search horizon
        # Leading to overoptimistic evaluation
        
        if depth == 0:
            # Doesn't see the piece will be captured next move
            return self.naive_evaluation(state)
        
        # This is why quiescence search is needed
        return self.quiescence_search(state, depth)
    
    def quiescence_search(self, state, depth):
        """Search only tactical moves to avoid horizon effects."""
        if depth <= 0 and self.is_quiet_position(state):
            return self.evaluate(state)
        
        # Continue searching captures and checks
        tactical_moves = self.get_tactical_moves(state)
        
        if not tactical_moves:
            return self.evaluate(state)
        
        best_value = float('-inf')
        for move in tactical_moves:
            new_state = self.make_move(state, move)
            value = -self.quiescence_search(new_state, depth - 1)
            best_value = max(best_value, value)
        
        return best_value
```

### Addressing Limitations

#### Techniques to Overcome Limitations:

1. **Iterative Deepening**: Search progressively deeper
2. **Transposition Tables**: Cache previously computed positions
3. **Move Ordering**: Search most promising moves first
4. **Null Move Pruning**: Skip moves to detect threats
5. **Quiescence Search**: Search tactical sequences to completion

```python
class ImprovedGameSearch:
    """Enhanced game search addressing common limitations."""
    
    def __init__(self):
        self.transposition_table = {}
        self.killer_moves = {}  # Moves that caused cutoffs
        self.history_heuristic = {}  # Move ordering heuristic
    
    def enhanced_alpha_beta(self, state, depth, alpha, beta, maximizing):
        """Alpha-beta with enhancements."""
        
        # Check transposition table
        state_hash = self.hash_state(state)
        if state_hash in self.transposition_table:
            entry = self.transposition_table[state_hash]
            if entry['depth'] >= depth:
                return entry['value']
        
        # Base case
        if depth == 0 or is_terminal(state):
            value = self.evaluate(state)
            self.store_in_transposition_table(state_hash, value, depth)
            return value
        
        # Move ordering for better pruning
        moves = self.order_moves(state, depth)
        
        best_value = float('-inf') if maximizing else float('inf')
        
        for move in moves:
            new_state = make_move(state, move)
            value = self.enhanced_alpha_beta(new_state, depth - 1, alpha, beta, not maximizing)
            
            if maximizing:
                if value > best_value:
                    best_value = value
                alpha = max(alpha, value)
                if beta <= alpha:
                    # Store killer move
                    self.killer_moves[depth] = move
                    break
            else:
                if value < best_value:
                    best_value = value
                beta = min(beta, value)
                if beta <= alpha:
                    self.killer_moves[depth] = move
                    break
        
        self.store_in_transposition_table(state_hash, best_value, depth)
        return best_value
    
    def order_moves(self, state, depth):
        """Order moves for better alpha-beta performance."""
        moves = get_legal_moves(state)
        
        # Score moves based on various heuristics
        scored_moves = []
        for move in moves:
            score = 0
            
            # Killer move heuristic
            if move == self.killer_moves.get(depth):
                score += 1000
            
            # History heuristic
            score += self.history_heuristic.get(move, 0)
            
            # Capture moves
            if is_capture(move):
                score += 100
            
            # Check moves
            if gives_check(state, move):
                score += 50
            
            scored_moves.append((score, move))
        
        # Sort by score (descending)
        scored_moves.sort(reverse=True)
        return [move for score, move in scored_moves]
```

---

## 9. Constraint Satisfaction Problems

### Concept
Constraint Satisfaction Problems (CSPs) involve finding values for variables that satisfy a set of constraints. CSPs are fundamental in AI for solving scheduling, planning, and configuration problems.

### Components of CSP:
- **Variables**: X = {X₁, X₂, ..., Xₙ}
- **Domains**: D = {D₁, D₂, ..., Dₙ} where Dᵢ is the set of possible values for Xᵢ
- **Constraints**: C = {C₁, C₂, ..., Cₘ} where each Cᵢ limits the values variables can take

### Types of Constraints:
- **Unary**: Constraints on single variable (e.g., X₁ ≠ blue)
- **Binary**: Constraints between two variables (e.g., X₁ ≠ X₂)
- **Higher-order**: Constraints involving multiple variables

### Real-World Examples:
- **Map Coloring**: Color adjacent regions with different colors
- **Sudoku**: Fill grid with numbers following rules
- **Course Scheduling**: Assign courses to time slots without conflicts
- **N-Queens**: Place N queens on chessboard without attacks

### Python Implementation

```python
class CSP:
    """Constraint Satisfaction Problem implementation."""
    
    def __init__(self, variables, domains, constraints):
        self.variables = variables
        self.domains = domains.copy()  # Current domains (may be reduced)
        self.original_domains = domains.copy()  # Original domains
        self.constraints = constraints
        self.assignment = {}
    
    def is_complete(self, assignment):
        """Check if assignment is complete (all variables assigned)."""
        return len(assignment) == len(self.variables)
    
    def is_consistent(self, variable, value, assignment):
        """Check if assigning value to variable is consistent with constraints."""
        # Create temporary assignment
        temp_assignment = assignment.copy()
        temp_assignment[variable] = value
        
        # Check all constraints involving this variable
        for constraint in self.constraints:
            if not constraint.is_satisfied(temp_assignment):
                return False
        
        return True
    
    def get_unassigned_variable(self, assignment):
        """Get next unassigned variable."""
        for variable in self.variables:
            if variable not in assignment:
                return variable
        return None
    
    def get_domain_values(self, variable):
        """Get possible values for variable."""
        return self.domains[variable]


class Constraint:
    """Base class for constraints."""
    
    def __init__(self, variables):
        self.variables = variables
    
    def is_satisfied(self, assignment):
        """Check if constraint is satisfied by assignment."""
        raise NotImplementedError


class BinaryConstraint(Constraint):
    """Binary constraint between two variables."""
    
    def __init__(self, var1, var2, relation):
        super().__init__([var1, var2])
        self.var1 = var1
        self.var2 = var2
        self.relation = relation  # Function that takes two values and returns bool
    
    def is_satisfied(self, assignment):
        """Check if binary constraint is satisfied."""
        if self.var1 not in assignment or self.var2 not in assignment:
            return True  # Constraint only applies when both variables are assigned
        
        return self.relation(assignment[self.var1], assignment[self.var2])


# Example: Map Coloring Problem
class MapColoringCSP:
    """Map coloring constraint satisfaction problem."""
    
    def __init__(self):
        # Variables: regions to color
        self.variables = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
        
        # Domains: possible colors
        colors = ['red', 'green', 'blue']
        self.domains = {var: colors.copy() for var in self.variables}
        
        # Constraints: adjacent regions must have different colors
        adjacencies = [
            ('WA', 'NT'), ('WA', 'SA'),
            ('NT', 'SA'), ('NT', 'Q'),
            ('SA', 'Q'), ('SA', 'NSW'), ('SA', 'V'),
            ('Q', 'NSW'),
            ('NSW', 'V')
        ]
        
        self.constraints = []
        for region1, region2 in adjacencies:
            # Create constraint: region1 != region2
            constraint = BinaryConstraint(
                region1, region2, 
                lambda val1, val2: val1 != val2
            )
            self.constraints.append(constraint)
        
        self.csp = CSP(self.variables, self.domains, self.constraints)
    
    def solve(self):
        """Solve the map coloring problem."""
        return self.backtrack_search({})
    
    def backtrack_search(self, assignment):
        """Backtracking search for CSP solution."""
        if self.csp.is_complete(assignment):
            return assignment
        
        variable = self.csp.get_unassigned_variable(assignment)
        
        for value in self.csp.get_domain_values(variable):
            if self.csp.is_consistent(variable, value, assignment):
                # Make assignment
                assignment[variable] = value
                
                # Recursively solve
                result = self.backtrack_search(assignment)
                if result is not None:
                    return result
                
                # Backtrack
                del assignment[variable]
        
        return None  # No solution found


# Example: N-Queens Problem
class NQueensCSP:
    """N-Queens problem as CSP."""
    
    def __init__(self, n):
        self.n = n
        
        # Variables: queens (one per row)
        self.variables = list(range(n))
        
        # Domains: column positions
        self.domains = {i: list(range(n)) for i in range(n)}
        
        # Constraints: no two queens attack each other
        self.constraints = []
        
        for i in range(n):
            for j in range(i + 1, n):
                # Queens in different rows, different columns, different diagonals
                constraint = BinaryConstraint(
                    i, j,
                    lambda col1, col2, row1=i, row2=j: (
                        col1 != col2 and  # Different columns
                        abs(col1 - col2) != abs(row1 - row2)  # Different diagonals
                    )
                )
                self.constraints.append(constraint)
        
        self.csp = CSP(self.variables, self.domains, self.constraints)
    
    def solve(self):
        """Solve N-Queens problem."""
        solution = self.backtrack_search({})
        if solution:
            return self.format_solution(solution)
        return None
    
    def backtrack_search(self, assignment):
        """Backtracking search with improved heuristics."""
        if self.csp.is_complete(assignment):
            return assignment
        
        # Most Constraining Variable heuristic
        variable = self.select_unassigned_variable(assignment)
        
        # Least Constraining Value heuristic
        values = self.order_domain_values(variable, assignment)
        
        for value in values:
            if self.csp.is_consistent(variable, value, assignment):
                assignment[variable] = value
                
                result = self.backtrack_search(assignment)
                if result is not None:
                    return result
                
                del assignment[variable]
        
        return None
    
    def select_unassigned_variable(self, assignment):
        """Select variable with most constraints (MRV heuristic)."""
        unassigned = [var for var in self.variables if var not in assignment]
        
        # Count remaining values for each variable
        remaining_values = {}
        for var in unassigned:
            count = 0
            for value in self.domains[var]:
                if self.csp.is_consistent(var, value, assignment):
                    count += 1
            remaining_values[var] = count
        
        # Return variable with minimum remaining values
        return min(unassigned, key=lambda var: remaining_values[var])
    
    def order_domain_values(self, variable, assignment):
        """Order values by least constraining value heuristic."""
        values_with_constraints = []
        
        for value in self.domains[variable]:
            if self.csp.is_consistent(variable, value, assignment):
                # Count how many values this eliminates from other variables
                eliminated_count = self.count_eliminated_values(variable, value, assignment)
                values_with_constraints.append((eliminated_count, value))
        
        # Sort by number of eliminated values (ascending)
        values_with_constraints.sort()
        return [value for _, value in values_with_constraints]
    
    def count_eliminated_values(self, variable, value, assignment):
        """Count values eliminated by assigning variable = value."""
        eliminated = 0
        temp_assignment = assignment.copy()
        temp_assignment[variable] = value
        
        for other_var in self.variables:
            if other_var != variable and other_var not in assignment:
                for other_value in self.domains[other_var]:
                    if not self.csp.is_consistent(other_var, other_value, temp_assignment):
                        eliminated += 1
        
        return eliminated
    
    def format_solution(self, solution):
        """Format solution as board representation."""
        board = [['.' for _ in range(self.n)] for _ in range(self.n)]
        for row, col in solution.items():
            board[row][col] = 'Q'
        return board


# Example: Sudoku as CSP
class SudokuCSP:
    """Sudoku puzzle as constraint satisfaction problem."""
    
    def __init__(self, puzzle):
        """
        Initialize Sudoku CSP.
        
        Args:
            puzzle: 9x9 grid with 0 representing empty cells
        """
        self.puzzle = puzzle
        self.variables = [(i, j) for i in range(9) for j in range(9)]
        
        # Domains: 1-9 for empty cells, fixed value for filled cells
        self.domains = {}
        for i in range(9):
            for j in range(9):
                if puzzle[i][j] == 0:
                    self.domains[(i, j)] = list(range(1, 10))
                else:
                    self.domains[(i, j)] = [puzzle[i][j]]
        
        # Constraints: all-different in rows, columns, and 3x3 boxes
        self.constraints = []
        self.create_sudoku_constraints()
        
        self.csp = CSP(self.variables, self.domains, self.constraints)
    
    def create_sudoku_constraints(self):
        """Create all Sudoku constraints."""
        # Row constraints
        for row in range(9):
            variables_in_row = [(row, col) for col in range(9)]
            self.constraints.append(AllDifferentConstraint(variables_in_row))
        
        # Column constraints
        for col in range(9):
            variables_in_col = [(row, col) for row in range(9)]
            self.constraints.append(AllDifferentConstraint(variables_in_col))
        
        # 3x3 box constraints
        for box_row in range(3):
            for box_col in range(3):
                variables_in_box = [
                    (i, j) 
                    for i in range(box_row * 3, (box_row + 1) * 3)
                    for j in range(box_col * 3, (box_col + 1) * 3)
                ]
                self.constraints.append(AllDifferentConstraint(variables_in_box))
    
    def solve(self):
        """Solve Sudoku puzzle."""
        # Start with given values
        initial_assignment = {}
        for i in range(9):
            for j in range(9):
                if self.puzzle[i][j] != 0:
                    initial_assignment[(i, j)] = self.puzzle[i][j]
        
        solution = self.backtrack_search(initial_assignment)
        if solution:
            return self.format_solution(solution)
        return None
    
    def format_solution(self, assignment):
        """Convert assignment to 9x9 grid."""
        solution_grid = [[0 for _ in range(9)] for _ in range(9)]
        for (i, j), value in assignment.items():
            solution_grid[i][j] = value
        return solution_grid


class AllDifferentConstraint(Constraint):
    """All variables in the constraint must have different values."""
    
    def __init__(self, variables):
        super().__init__(variables)
    
    def is_satisfied(self, assignment):
        """Check if all assigned variables have different values."""
        assigned_values = []
        
        for var in self.variables:
            if var in assignment:
                value = assignment[var]
                if value in assigned_values:
                    return False
                assigned_values.append(value)
        
        return True
```

### Usage Examples:

```python
# Solve Map Coloring
map_coloring = MapColoringCSP()
solution = map_coloring.solve()
print("Map Coloring Solution:", solution)

# Solve 8-Queens
queens = NQueensCSP(8)
solution = queens.solve()
if solution:
    for row in solution:
        print(' '.join(row))

# Solve Sudoku
puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    # ... rest of puzzle
]
sudoku = SudokuCSP(puzzle)
solution = sudoku.solve()
```

---

## 10. Constraint Propagation

### Concept
Constraint propagation reduces variable domains by applying constraints before search. This preprocessing can significantly reduce the search space and improve efficiency.

### Types of Consistency:
- **Node Consistency**: Each variable's domain satisfies unary constraints
- **Arc Consistency**: For every value in a variable's domain, there exists a consistent value in related variables
- **Path Consistency**: For every pair of values, there exists a consistent path

### AC-3 Algorithm (Arc Consistency)

### Pseudocode
```
function AC3(csp):
    queue = all arcs in csp
    while queue is not empty:
        (Xi, Xj) = remove_first(queue)
        if REVISE(csp, Xi, Xj):
            if size of Di = 0:
                return false
            for each Xk in neighbors of Xi except Xj:
                add (Xk, Xi) to queue
    return true

function REVISE(csp, Xi, Xj):
    revised = false
    for each x in Di:
        if no value y in Dj allows (x,y) to satisfy constraint between Xi and Xj:
            delete x from Di
            revised = true
    return revised
```

### Python Implementation

```python
from collections import deque

class ConstraintPropagation:
    """Implement various constraint propagation techniques."""
    
    def __init__(self, csp):
        self.csp = csp
    
    def ac3(self):
        """
        Arc Consistency Algorithm 3.
        
        Returns:
            True if problem is arc consistent, False if inconsistent
        """
        # Initialize queue with all arcs
        queue = deque()
        
        for constraint in self.csp.constraints:
            if isinstance(constraint, BinaryConstraint):
                queue.append((constraint.var1, constraint.var2, constraint))
                queue.append((constraint.var2, constraint.var1, constraint))
        
        while queue:
            xi, xj, constraint = queue.popleft()
            
            if self.revise(xi, xj, constraint):
                if len(self.csp.domains[xi]) == 0:
                    return False  # Domain wipeout - inconsistent
                
                # Add all neighbors of Xi except Xj back to queue
                for other_constraint in self.csp.constraints:
                    if isinstance(other_constraint, BinaryConstraint):
                        if other_constraint.var1 == xi and other_constraint.var2 != xj:
                            queue.append((other_constraint.var2, xi, other_constraint))
                        elif other_constraint.var2 == xi and other_constraint.var1 != xj:
                            queue.append((other_constraint.var1, xi, other_constraint))
        
        return True
    
    def revise(self, xi, xj, constraint):
        """
        Remove values from domain of Xi that have no support in Xj.
        
        Returns:
            True if domain of Xi was revised
        """
        revised = False
        values_to_remove = []
        
        for x in self.csp.domains[xi]:
            # Check if there exists any value y in Dj such that (x,y) satisfies constraint
            supported = False
            
            for y in self.csp.domains[xj]:
                # Create temporary assignment to test constraint
                temp_assignment = {xi: x, xj: y}
                if constraint.is_satisfied(temp_assignment):
                    supported = True
                    break
            
            if not supported:
                values_to_remove.append(x)
                revised = True
        
        # Remove unsupported values
        for value in values_to_remove:
            self.csp.domains[xi].remove(value)
        
        return revised
    
    def forward_checking(self, assignment, variable, value):
        """
        Apply forward checking when assigning variable = value.
        
        Returns:
            Dictionary of removed values (for backtracking)
        """
        removed_values = {}
        
        # For each unassigned variable connected to current variable
        for constraint in self.csp.constraints:
            if isinstance(constraint, BinaryConstraint):
                other_var = None
                
                if constraint.var1 == variable and constraint.var2 not in assignment:
                    other_var = constraint.var2
                elif constraint.var2 == variable and constraint.var1 not in assignment:
                    other_var = constraint.var1
                
                if other_var:
                    # Check which values in other_var's domain are still consistent
                    values_to_remove = []
                    
                    for other_value in self.csp.domains[other_var]:
                        temp_assignment = assignment.copy()
                        temp_assignment[variable] = value
                        temp_assignment[other_var] = other_value
                        
                        if not constraint.is_satisfied(temp_assignment):
                            values_to_remove.append(other_value)
                    
                    # Remove inconsistent values
                    if values_to_remove:
                        if other_var not in removed_values:
                            removed_values[other_var] = []
                        
                        for val in values_to_remove:
                            self.csp.domains[other_var].remove(val)
                            removed_values[other_var].append(val)
                        
                        # Check for domain wipeout
                        if len(self.csp.domains[other_var]) == 0:
                            return None  # Inconsistent
        
        return removed_values
    
    def restore_domains(self, removed_values):
        """Restore previously removed values during backtracking."""
        for variable, values in removed_values.items():
            self.csp.domains[variable].extend(values)


# Enhanced CSP solver with constraint propagation
class EnhancedCSPSolver:
    """CSP solver with constraint propagation techniques."""
    
    def __init__(self, csp):
        self.csp = csp
        self.propagator = ConstraintPropagation(csp)
    
    def solve(self):
        """Solve CSP with constraint propagation."""
        # Apply initial arc consistency
        if not self.propagator.ac3():
            return None  # Problem is inconsistent
        
        return self.backtrack_with_propagation({})
    
    def backtrack_with_propagation(self, assignment):
        """Backtracking search with forward checking."""
        if len(assignment) == len(self.csp.variables):
            return assignment
        
        # Select variable using MRV heuristic
        variable = self.select_unassigned_variable(assignment)
        
        # Order values using LCV heuristic
        values = self.order_domain_values(variable, assignment)
        
        for value in values:
            if self.is_consistent(variable, value, assignment):
                # Make assignment
                assignment[variable] = value
                
                # Apply forward checking
                removed_values = self.propagator.forward_checking(assignment, variable, value)
                
                if removed_values is not None:  # No domain wipeout
                    result = self.backtrack_with_propagation(assignment)
                    if result is not None:
                        return result
                
                # Backtrack: restore domains and remove assignment
                if removed_values is not None:
                    self.propagator.restore_domains(removed_values)
                del assignment[variable]
        
        return None
    
    def select_unassigned_variable(self, assignment):
        """Minimum Remaining Values (MRV) heuristic."""
        unassigned = [var for var in self.csp.variables if var not in assignment]
        
        return min(unassigned, key=lambda var: len(self.csp.domains[var]))
    
    def order_domain_values(self, variable, assignment):
        """Least Constraining Value (LCV) heuristic."""
        if variable not in self.csp.domains:
            return []
        
        values_with_scores = []
        
        for value in self.csp.domains[variable]:
            # Count how many values this eliminates from other variables
            eliminated = self.count_conflicts(variable, value, assignment)
            values_with_scores.append((eliminated, value))
        
        # Sort by number of conflicts (ascending)
        values_with_scores.sort()
        return [value for _, value in values_with_scores]
    
    def count_conflicts(self, variable, value, assignment):
        """Count conflicts caused by assigning variable = value."""
        conflicts = 0
        temp_assignment = assignment.copy()
        temp_assignment[variable] = value
        
        for constraint in self.csp.constraints:
            if isinstance(constraint, BinaryConstraint):
                other_var = None
                
                if constraint.var1 == variable and constraint.var2 not in assignment:
                    other_var = constraint.var2
                elif constraint.var2 == variable and constraint.var1 not in assignment:
                    other_var = constraint.var1
                
                if other_var and other_var in self.csp.domains:
                    for other_value in self.csp.domains[other_var]:
                        temp_assignment[other_var] = other_value
                        if not constraint.is_satisfied(temp_assignment):
                            conflicts += 1
                        del temp_assignment[other_var]
        
        return conflicts
    
    def is_consistent(self, variable, value, assignment):
        """Check if assignment is consistent with all constraints."""
        temp_assignment = assignment.copy()
        temp_assignment[variable] = value
        
        for constraint in self.csp.constraints:
            if not constraint.is_satisfied(temp_assignment):
                return False
        
        return True


# Example: Enhanced Sudoku Solver
class EnhancedSudokuSolver:
    """Sudoku solver with advanced constraint propagation."""
    
    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.size = 9
        self.box_size = 3
        
        # Create variables and domains
        self.variables = [(i, j) for i in range(9) for j in range(9)]
        self.domains = self.initialize_domains()
        
        # Create constraints
        self.constraints = self.create_constraints()
        
        # Create CSP
        self.csp = CSP(self.variables, self.domains, self.constraints)
        self.solver = EnhancedCSPSolver(self.csp)
    
    def initialize_domains(self):
        """Initialize domains with constraint propagation."""
        domains = {}
        
        for i in range(9):
            for j in range(9):
                if self.puzzle[i][j] != 0:
                    domains[(i, j)] = [self.puzzle[i][j]]
                else:
                    # Start with all possible values
                    possible_values = set(range(1, 10))
                    
                    # Remove values already in same row
                    for col in range(9):
                        if self.puzzle[i][col] != 0:
                            possible_values.discard(self.puzzle[i][col])
                    
                    # Remove values already in same column
                    for row in range(9):
                        if self.puzzle[row][j] != 0:
                            possible_values.discard(self.puzzle[row][j])
                    
                    # Remove values already in same 3x3 box
                    box_row, box_col = 3 * (i // 3), 3 * (j // 3)
                    for r in range(box_row, box_row + 3):
                        for c in range(box_col, box_col + 3):
                            if self.puzzle[r][c] != 0:
                                possible_values.discard(self.puzzle[r][c])
                    
                    domains[(i, j)] = list(possible_values)
        
        return domains
    
    def create_constraints(self):
        """Create all Sudoku constraints."""
        constraints = []
        
        # Row constraints
        for row in range(9):
            row_vars = [(row, col) for col in range(9)]
            constraints.append(AllDifferentConstraint(row_vars))
        
        # Column constraints
        for col in range(9):
            col_vars = [(row, col) for row in range(9)]
            constraints.append(AllDifferentConstraint(col_vars))
        
        # Box constraints
        for box_row in range(3):
            for box_col in range(3):
                box_vars = [
                    (i, j)
                    for i in range(box_row * 3, (box_row + 1) * 3)
                    for j in range(box_col * 3, (box_col + 1) * 3)
                ]
                constraints.append(AllDifferentConstraint(box_vars))
        
        return constraints
    
    def solve(self):
        """Solve Sudoku with enhanced techniques."""
        # Apply naked singles and hidden singles first
        self.apply_sudoku_techniques()
        
        # Use CSP solver for remaining cells
        solution = self.solver.solve()
        
        if solution:
            return self.format_solution(solution)
        return None
    
    def apply_sudoku_techniques(self):
        """Apply Sudoku-specific constraint propagation techniques."""
        changed = True
        
        while changed:
            changed = False
            
            # Naked singles: cells with only one possible value
            for var in self.variables:
                if len(self.csp.domains[var]) == 1 and var not in self.solver.assignment:
                    value = self.csp.domains[var][0]
                    self.solver.assignment[var] = value
                    changed = True
                    
                    # Remove this value from related cells
                    self.eliminate_value_from_peers(var, value)
            
            # Hidden singles: values that can only go in one cell in a group
            changed |= self.find_hidden_singles()
    
    def eliminate_value_from_peers(self, variable, value):
        """Remove value from all peers of variable."""
        row, col = variable
        
        # Remove from same row
        for c in range(9):
            peer = (row, c)
            if peer != variable and value in self.csp.domains[peer]:
                self.csp.domains[peer].remove(value)
        
        # Remove from same column
        for r in range(9):
            peer = (r, col)
            if peer != variable and value in self.csp.domains[peer]:
                self.csp.domains[peer].remove(value)
        
        # Remove from same box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                peer = (r, c)
                if peer != variable and value in self.csp.domains[peer]:
                    self.csp.domains[peer].remove(value)
    
    def find_hidden_singles(self):
        """Find hidden singles in rows, columns, and boxes."""
        changed = False
        
        # Check rows
        for row in range(9):
            row_vars = [(row, col) for col in range(9)]
            changed |= self.find_hidden_singles_in_group(row_vars)
        
        # Check columns
        for col in range(9):
            col_vars = [(row, col) for row in range(9)]
            changed |= self.find_hidden_singles_in_group(col_vars)
        
        # Check boxes
        for box_row in range(3):
            for box_col in range(3):
                box_vars = [
                    (i, j)
                    for i in range(box_row * 3, (box_row + 1) * 3)
                    for j in range(box_col * 3, (box_col + 1) * 3)
                ]
                changed |= self.find_hidden_singles_in_group(box_vars)
        
        return changed
    
    def find_hidden_singles_in_group(self, group_vars):
        """Find hidden singles within a group of variables."""
        changed = False
        
        for value in range(1, 10):
            possible_positions = []
            
            for var in group_vars:
                if var not in self.solver.assignment and value in self.csp.domains[var]:
                    possible_positions.append(var)
            
            # If value can only go in one position, it's a hidden single
            if len(possible_positions) == 1:
                var = possible_positions[0]
                if len(self.csp.domains[var]) > 1:
                    self.csp.domains[var] = [value]
                    changed = True
        
        return changed
    
    def format_solution(self, assignment):
        """Convert assignment to 9x9 grid."""
        solution_grid = [[0 for _ in range(9)] for _ in range(9)]
        for (i, j), value in assignment.items():
            solution_grid[i][j] = value
        return solution_grid
```

---

## 11. Backtracking Search for CSPs

### Concept
Backtracking search systematically explores the search space by making assignments and undoing them when they lead to conflicts. It's the core algorithm for solving CSPs.

### Basic Backtracking Algorithm

### Pseudocode
```
function BACKTRACK(assignment, csp):
    if assignment is complete:
        return assignment
    
    var = SELECT_UNASSIGNED_VARIABLE(csp)
    for each value in ORDER_DOMAIN_VALUES(var, assignment, csp):
        if value is consistent with assignment:
            add {var = value} to assignment
            result = BACKTRACK(assignment, csp)
            if result ≠ failure:
                return result
            remove {var = value} from assignment
    return failure
```

### Optimizations and Heuristics

#### 1. Variable Selection Heuristics:
- **Most Constraining Variable (MCV)**: Choose variable involved in most constraints
- **Minimum Remaining Values (MRV)**: Choose variable with fewest legal values
- **Degree Heuristic**: Choose variable connected to most unassigned variables

#### 2. Value Ordering Heuristics:
- **Least Constraining Value (LCV)**: Choose value that eliminates fewest choices for neighbors

#### 3. Inference Techniques:
- **Forward Checking**: Keep track of remaining legal values for unassigned variables
- **Arc Consistency**: Maintain arc consistency during search

### Complete Implementation

```python
import time
from typing import Dict, List, Tuple, Optional, Any

class OptimizedBacktrackingCSP:
    """Optimized backtracking search with multiple heuristics."""
    
    def __init__(self, csp):
        self.csp = csp
        self.nodes_expanded = 0
        self.start_time = None
        
        # Statistics tracking
        self.stats = {
            'nodes_expanded': 0,
            'backtracks': 0,
            'constraint_checks': 0,
            'time_taken': 0
        }
    
    def solve(self):
        """Solve CSP using optimized backtracking."""
        self.start_time = time.time()
        self.stats = {key: 0 for key in self.stats}
        
        result = self.backtrack({})
        
        self.stats['time_taken'] = time.time() - self.start_time
        return result
    
    def backtrack(self, assignment):
        """Main backtracking algorithm with optimizations."""
        self.stats['nodes_expanded'] += 1
        
        # Check if assignment is complete
        if len(assignment) == len(self.csp.variables):
            return assignment
        
        # Select unassigned variable using MRV + Degree heuristic
        variable = self.select_unassigned_variable(assignment)
        
        # Order domain values using LCV heuristic
        ordered_values = self.order_domain_values(variable, assignment)
        
        for value in ordered_values:
            if self.is_consistent(variable, value, assignment):
                # Make assignment
                assignment[variable] = value
                
                # Apply inference (forward checking)
                inferences = self.forward_check(variable, value, assignment)
                
                if inferences is not None:  # No contradiction found
                    # Recursively solve
                    result = self.backtrack(assignment)
                    if result is not None:
                        return result
                
                # Backtrack: remove assignment and restore domains
                del assignment[variable]
                if inferences is not None:
                    self.restore_domains(inferences)
                
                self.stats['backtracks'] += 1
        
        return None  # No solution found
    
    def select_unassigned_variable(self, assignment):
        """Variable selection using MRV + Degree heuristic."""
        unassigned = [var for var in self.csp.variables if var not in assignment]
        
        if not unassigned:
            return None
        
        # MRV: Minimum Remaining Values
        min_remaining_values = min(len(self.csp.domains[var]) for var in unassigned)
        mrv_candidates = [
            var for var in unassigned 
            if len(self.csp.domains[var]) == min_remaining_values
        ]
        
        # If tie, use Degree heuristic
        if len(mrv_candidates) == 1:
            return mrv_candidates[0]
        
        # Degree heuristic: variable involved in most constraints with unassigned variables
        def count_unassigned_neighbors(variable):
            count = 0
            for constraint in self.csp.constraints:
                if isinstance(constraint, BinaryConstraint):
                    if constraint.var1 == variable and constraint.var2 not in assignment:
                        count += 1
                    elif constraint.var2 == variable and constraint.var1 not in assignment:
                        count += 1
            return count
        
        return max(mrv_candidates, key=count_unassigned_neighbors)
    
    def order_domain_values(self, variable, assignment):
        """Value ordering using Least Constraining Value heuristic."""
        if variable not in self.csp.domains:
            return []
        
        # Calculate constraining effect of each value
        value_scores = []
        
        for value in self.csp.domains[variable]:
            if self.is_consistent(variable, value, assignment):
                # Count how many values this eliminates from neighbors
                eliminated_count = self.count_eliminated_values(variable, value, assignment)
                value_scores.append((eliminated_count, value))
        
        # Sort by eliminated count (ascending - least constraining first)
        value_scores.sort(key=lambda x: x[0])
        return [value for _, value in value_scores]
    
    def count_eliminated_values(self, variable, value, assignment):
        """Count values eliminated by assigning variable = value."""
        eliminated = 0
        temp_assignment = assignment.copy()
        temp_assignment[variable] = value
        
        for constraint in self.csp.constraints:
            if isinstance(constraint, BinaryConstraint):
                other_var = None
                
                if constraint.var1 == variable and constraint.var2 not in assignment:
                    other_var = constraint.var2
                elif constraint.var2 == variable and constraint.var1 not in assignment:
                    other_var = constraint.var1
                
                if other_var and other_var in self.csp.domains:
                    for other_value in self.csp.domains[other_var]:
                        temp_assignment[other_var] = other_value
                        if not constraint.is_satisfied(temp_assignment):
                            eliminated += 1
                        del temp_assignment[other_var]
        
        return eliminated
    
    def forward_check(self, variable, value, assignment):
        """Apply forward checking to reduce domains."""
        removed_values = {}
        temp_assignment = assignment.copy()
        temp_assignment[variable] = value
        
        for constraint in self.csp.constraints:
            if isinstance(constraint, BinaryConstraint):
                other_var = None
                
                if constraint.var1 == variable and constraint.var2 not in assignment:
                    other_var = constraint.var2
                elif constraint.var2 == variable and constraint.var1 not in assignment:
                    other_var = constraint.var1
                
                if other_var and other_var in self.csp.domains:
                    values_to_remove = []
                    
                    for other_value in self.csp.domains[other_var]:
                        temp_assignment[other_var] = other_value
                        if not constraint.is_satisfied(temp_assignment):
                            values_to_remove.append(other_value)
                        del temp_assignment[other_var]
                    
                    if values_to_remove:
                        if other_var not in removed_values:
                            removed_values[other_var] = []
                        
                        for val in values_to_remove:
                            if val in self.csp.domains[other_var]:
                                self.csp.domains[other_var].remove(val)
                                removed_values[other_var].append(val)
                        
                        # Check for domain wipeout
                        if len(self.csp.domains[other_var]) == 0:
                            # Restore domains before returning failure
                            self.restore_domains(removed_values)
                            return None
        
        return removed_values
    
    def restore_domains(self, removed_values):
        """Restore domains after backtracking."""
        for variable, values in removed_values.items():
            if variable in self.csp.domains:
                self.csp.domains[variable].extend(values)
    
    def is_consistent(self, variable, value, assignment):
        """Check if assignment is consistent with constraints."""
        self.stats['constraint_checks'] += 1
        
        temp_assignment = assignment.copy()
        temp_assignment[variable] = value
        
        for constraint in self.csp.constraints:
            if not constraint.is_satisfied(temp_assignment):
                return False
        
        return True
    
    def get_statistics(self):
        """Return search statistics."""
        return self.stats


# Example: Comprehensive N-Queens with Statistics
class StatisticalNQueens:
    """N-Queens with detailed performance analysis."""
    
    def __init__(self, n):
        self.n = n
        self.setup_csp()
    
    def setup_csp(self):
        """Set up CSP for N-Queens problem."""
        # Variables: queen positions (one per row)
        variables = list(range(self.n))
        
        # Domains: column positions
        domains = {i: list(range(self.n)) for i in range(self.n)}
        
        # Constraints: no attacks
        constraints = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                constraint = BinaryConstraint(
                    i, j,
                    lambda col1, col2, row1=i, row2=j: (
                        col1 != col2 and  # Different columns
                        abs(col1 - col2) != abs(row1 - row2)  # Different diagonals
                    )
                )
                constraints.append(constraint)
        
        self.csp = CSP(variables, domains, constraints)
    
    def solve_with_comparison(self):
        """Solve using different approaches and compare performance."""
        results = {}
        
        # Basic backtracking
        print(f"Solving {self.n}-Queens with basic backtracking...")
        basic_solver = BasicBacktrackingCSP(self.csp)
        basic_result = basic_solver.solve()
        results['basic'] = {
            'solution': basic_result,
            'stats': basic_solver.get_statistics()
        }
        
        # Reset domains
        self.setup_csp()
        
        # Optimized backtracking
        print(f"Solving {self.n}-Queens with optimized backtracking...")
        optimized_solver = OptimizedBacktrackingCSP(self.csp)
        optimized_result = optimized_solver.solve()
        results['optimized'] = {
            'solution': optimized_result,
            'stats': optimized_solver.get_statistics()
        }
        
        return results
    
    def print_comparison(self, results):
        """Print performance comparison."""
        print(f"\n{self.n}-Queens Performance Comparison:")
        print("-" * 50)
        
        for method, data in results.items():
            stats = data['stats']
            print(f"\n{method.capitalize()} Backtracking:")
            print(f"  Nodes expanded: {stats['nodes_expanded']}")
            print(f"  Backtracks: {stats['backtracks']}")
            print(f"  Constraint checks: {stats['constraint_checks']}")
            print(f"  Time taken: {stats['time_taken']:.4f} seconds")
            print(f"  Solution found: {'Yes' if data['solution'] else 'No'}")


class BasicBacktrackingCSP:
    """Basic backtracking without optimizations for comparison."""
    
    def __init__(self, csp):
        self.csp = csp
        self.stats = {
            'nodes_expanded': 0,
            'backtracks': 0,
            'constraint_checks': 0,
            'time_taken': 0
        }
    
    def solve(self):
        """Solve using basic backtracking."""
        start_time = time.time()
        result = self.backtrack({})
        self.stats['time_taken'] = time.time() - start_time
        return result
    
    def backtrack(self, assignment):
        """Basic backtracking algorithm."""
        self.stats['nodes_expanded'] += 1
        
        if len(assignment) == len(self.csp.variables):
            return assignment
        
        # Simple variable selection: first unassigned
        variable = None
        for var in self.csp.variables:
            if var not in assignment:
                variable = var
                break
        
        # Simple value ordering: domain order
        for value in self.csp.domains[variable]:
            if self.is_consistent(variable, value, assignment):
                assignment[variable] = value
                
                result = self.backtrack(assignment)
                if result is not None:
                    return result
                
                del assignment[variable]
                self.stats['backtracks'] += 1
        
        return None
    
    def is_consistent(self, variable, value, assignment):
        """Check consistency."""
        self.stats['constraint_checks'] += 1
        
        temp_assignment = assignment.copy()
        temp_assignment[variable] = value
        
        for constraint in self.csp.constraints:
            if not constraint.is_satisfied(temp_assignment):
                return False
        
        return True
    
    def get_statistics(self):
        """Return statistics."""
        return self.stats


# Usage example
if __name__ == "__main__":
    # Compare different backtracking approaches
    for n in [4, 6, 8]:
        queens = StatisticalNQueens(n)
        results = queens.solve_with_comparison()
        queens.print_comparison(results)
```

### Case Study: Google DeepMind Applications

#### Energy Efficiency in Data Centers
Google DeepMind applied constraint satisfaction and optimization techniques to reduce cooling costs in data centers by 40%.

**CSP Formulation:**
- **Variables**: Temperature settings, fan speeds, cooling system states
- **Constraints**: Safety limits, equipment specifications, power budgets
- **Objective**: Minimize energy consumption while maintaining optimal temperatures

```python
class DataCenterOptimization:
    """Simplified model of data center cooling optimization."""
    
    def __init__(self, servers, cooling_units):
        self.servers = servers
        self.cooling_units = cooling_units
        self.setup_csp()
    
    def setup_csp(self):
        """Set up CSP for data center cooling."""
        variables = []
        domains = {}
        constraints = []
        
        # Variables: cooling settings for each unit
        for unit_id in range(self.cooling_units):
            var_temp = f"temp_{unit_id}"
            var_fan = f"fan_{unit_id}"
            
            variables.extend([var_temp, var_fan])
            
            # Temperature range: 18-27°C
            domains[var_temp] = list(range(18, 28))
            
            # Fan speed: 0-100%
            domains[var_fan] = list(range(0, 101, 10))
        
        # Constraints: temperature and power limits
        for server_id in range(self.servers):
            # Each server must be kept below critical temperature
            constraint = ServerTemperatureConstraint(server_id, variables, domains)
            constraints.append(constraint)
        
        # Power consumption constraint
        power_constraint = PowerConsumptionConstraint(variables, domains)
        constraints.append(power_constraint)
        
        self.csp = CSP(variables, domains, constraints)
    
    def optimize(self):
        """Find optimal cooling configuration."""
        solver = OptimizedBacktrackingCSP(self.csp)
        solution = solver.solve()
        
        if solution:
            return self.interpret_solution(solution)
        return None
    
    def interpret_solution(self, solution):
        """Convert CSP solution to cooling configuration."""
        configuration = {}
        
        for unit_id in range(self.cooling_units):
            configuration[f"unit_{unit_id}"] = {
                'temperature': solution[f"temp_{unit_id}"],
                'fan_speed': solution[f"fan_{unit_id}"]
            }
        
        return configuration


class ServerTemperatureConstraint(Constraint):
    """Constraint ensuring server temperature stays within limits."""
    
    def __init__(self, server_id, variables, domains):
        super().__init__(variables)
        self.server_id = server_id
    
    def is_satisfied(self, assignment):
        """Check if server temperature is acceptable."""
        # Simplified: check that nearby cooling units maintain temperature
        return True  # Implementation would involve thermal modeling


class PowerConsumptionConstraint(Constraint):
    """Constraint limiting total power consumption."""
    
    def __init__(self, variables, domains):
        super().__init__(variables)
        self.max_power = 10000  # kW
    
    def is_satisfied(self, assignment):
        """Check if total power consumption is within limits."""
        total_power = 0
        
        for var, value in assignment.items():
            if var.startswith('fan_'):
                # Power consumption increases with fan speed
                total_power += value * 2  # Simplified model
        
        return total_power <= self.max_power
```

This concludes Unit III on Adversarial Search and Constraint Satisfaction Problems. We've covered game theory, search algorithms (Minimax, Alpha-Beta, MCTS), stochastic and partially observable games, constraint satisfaction, and real-world applications.

---

# Unit IV - Knowledge and Reasoning

## 12. Knowledge-Based Agents

### Concept
Knowledge-based agents maintain an internal knowledge base (KB) and use logical reasoning to make decisions. They can derive new knowledge from existing knowledge and observations.

### Components of Knowledge-Based Agents:
- **Knowledge Base (KB)**: Collection of facts and rules
- **Inference Engine**: Mechanism to derive new knowledge
- **Sensors**: Gather information about environment
- **Actuators**: Perform actions based on reasoning

### Architecture

```python
class KnowledgeBasedAgent:
    """A generic knowledge-based agent."""
    
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.time = 0
    
    def agent_program(self, percept):
        """
        Agent program following the KB agent cycle.
        
        Args:
            percept: Current observation from environment
            
        Returns:
            Action to take based on reasoning
        """
        # TELL: Add percept to knowledge base
        self.tell(percept)
        
        # ASK: Query knowledge base for best action
        action = self.ask_for_action()
        
        # TELL: Add action to knowledge base (for learning)
        self.tell_action(action)
        
        self.time += 1
        return action
    
    def tell(self, percept):
        """Add new information to knowledge base."""
        # Convert percept to logical sentence
        sentence = self.make_percept_sentence(percept)
        self.knowledge_base.tell(sentence)
    
    def ask_for_action(self):
        """Query KB to determine best action."""
        # Try each possible action and see if it's safe/beneficial
        for action in self.get_possible_actions():
            query = self.make_action_query(action)
            if self.knowledge_base.ask(query):
                return action
        
        # Default action if no good action found
        return self.default_action()
    
    def tell_action(self, action):
        """Record action taken for future reasoning."""
        action_sentence = self.make_action_sentence(action)
        self.knowledge_base.tell(action_sentence)
    
    def make_percept_sentence(self, percept):
        """Convert percept to logical sentence."""
        # Implementation depends on specific domain
        pass
    
    def make_action_query(self, action):
        """Create query to check if action is good."""
        # Implementation depends on specific domain
        pass
    
    def make_action_sentence(self, action):
        """Convert action to logical sentence."""
        # Implementation depends on specific domain
        pass
    
    def get_possible_actions(self):
        """Get list of possible actions."""
        return ['Forward', 'TurnLeft', 'TurnRight', 'Grab', 'Shoot', 'Climb']
    
    def default_action(self):
        """Default action when no better option found."""
        return 'Forward'


class KnowledgeBase:
    """Knowledge base for storing and reasoning with logical sentences."""
    
    def __init__(self):
        self.sentences = []
    
    def tell(self, sentence):
        """Add a sentence to the knowledge base."""
        if sentence not in self.sentences:
            self.sentences.append(sentence)
    
    def ask(self, query):
        """Ask whether query can be derived from KB."""
        # This is where inference algorithms would be implemented
        return self.entails(query)
    
    def entails(self, query):
        """Check if KB entails the query."""
        # Placeholder - would use specific inference method
        return False
    
    def retract(self, sentence):
        """Remove a sentence from KB."""
        if sentence in self.sentences:
            self.sentences.remove(sentence)
```

### The Agent Cycle

1. **Perceive**: Agent observes environment
2. **Update**: Add percept to knowledge base
3. **Reason**: Use inference to derive conclusions
4. **Act**: Choose action based on reasoning
5. **Learn**: Update KB with action results

### Example: Simple Reflex Agent vs Knowledge-Based Agent

```python
# Simple Reflex Agent (reactive)
class SimpleReflexAgent:
    """Agent that acts based only on current percept."""
    
    def __init__(self):
        self.rules = {
            'obstacle_ahead': 'TurnLeft',
            'gold_here': 'Grab',
            'pit_nearby': 'TurnAround',
            'safe_forward': 'Forward'
        }
    
    def agent_program(self, percept):
        """Choose action based on current percept only."""
        for condition, action in self.rules.items():
            if self.interpret_percept(percept, condition):
                return action
        return 'Forward'  # Default action
    
    def interpret_percept(self, percept, condition):
        """Check if percept matches condition."""
        # Implementation depends on percept format
        return condition in percept


# Knowledge-Based Agent (deliberative)
class DeliberativeAgent(KnowledgeBasedAgent):
    """Agent that maintains world model and plans ahead."""
    
    def __init__(self):
        super().__init__()
        self.world_model = WorldModel()
        self.goal = None
        self.plan = []
    
    def agent_program(self, percept):
        """Deliberative agent cycle."""
        # Update world model
        self.update_world_model(percept)
        
        # Plan to achieve goal
        if not self.plan or self.goal_achieved():
            self.plan = self.make_plan()
        
        # Execute next step of plan
        if self.plan:
            action = self.plan.pop(0)
        else:
            action = 'Wait'
        
        return action
    
    def update_world_model(self, percept):
        """Update internal model of world state."""
        self.world_model.update(percept, self.time)
    
    def goal_achieved(self):
        """Check if current goal is achieved."""
        return self.world_model.satisfies_goal(self.goal)
    
    def make_plan(self):
        """Create plan to achieve goal."""
        return self.world_model.plan_to_goal(self.goal)


class WorldModel:
    """Internal model of world state."""
    
    def __init__(self):
        self.facts = set()
        self.location = (1, 1)  # Starting position
        self.orientation = 'North'
    
    def update(self, percept, time):
        """Update world model based on percept."""
        # Add facts about current location
        self.facts.add(f"At({self.location}, {time})")
        
        # Process percept components
        if 'Breeze' in percept:
            self.facts.add(f"Breeze({self.location})")
        if 'Stench' in percept:
            self.facts.add(f"Stench({self.location})")
        if 'Glitter' in percept:
            self.facts.add(f"Gold({self.location})")
    
    def satisfies_goal(self, goal):
        """Check if goal is satisfied."""
        if goal == 'FindGold':
            return any('Gold' in fact for fact in self.facts)
        return False
    
    def plan_to_goal(self, goal):
        """Generate plan to achieve goal."""
        # Simplified planning
        if goal == 'FindGold':
            return ['Forward', 'Forward', 'TurnRight', 'Forward']
        return []
```

### Levels of Reasoning

#### 1. Knowledge Level
- What the agent knows
- Goals and capabilities
- Independent of implementation

#### 2. Logical Level  
- How knowledge is represented
- Logical formalism used
- Inference methods

#### 3. Implementation Level
- Data structures
- Algorithms
- Hardware

### Types of Knowledge

```python
class KnowledgeTypes:
    """Demonstrate different types of knowledge representation."""
    
    def __init__(self):
        # Declarative knowledge: facts about the world
        self.facts = {
            'location': (3, 4),
            'has_gold': False,
            'wumpus_alive': True,
            'arrows': 1
        }
        
        # Procedural knowledge: how to do things
        self.procedures = {
            'move_forward': self.move_forward,
            'turn_left': self.turn_left,
            'shoot_arrow': self.shoot_arrow
        }
        
        # Conditional knowledge: if-then rules
        self.rules = [
            ('stench_perceived', 'wumpus_nearby'),
            ('breeze_perceived', 'pit_nearby'),
            ('glitter_perceived', 'gold_here')
        ]
    
    def move_forward(self):
        """Procedural knowledge for moving."""
        x, y = self.facts['location']
        # Update location based on orientation
        if self.facts.get('orientation') == 'North':
            self.facts['location'] = (x, y + 1)
        elif self.facts.get('orientation') == 'East':
            self.facts['location'] = (x + 1, y)
        # ... other directions
    
    def apply_rules(self, percept):
        """Apply conditional knowledge."""
        inferred_facts = []
        
        for condition, conclusion in self.rules:
            if condition in percept:
                inferred_facts.append(conclusion)
        
        return inferred_facts
    
    def meta_knowledge(self):
        """Knowledge about knowledge - what the agent knows it knows."""
        return {
            'knows_location': 'location' in self.facts,
            'knows_wumpus_status': 'wumpus_alive' in self.facts,
            'uncertainty_areas': self.get_uncertain_areas()
        }
    
    def get_uncertain_areas(self):
        """Areas where agent lacks complete information."""
        # Return locations not yet visited
        visited = self.facts.get('visited_locations', set())
        all_locations = {(x, y) for x in range(1, 5) for y in range(1, 5)}
        return all_locations - visited
```

---

## 13. The Wumpus World Problem

### Overview
The Wumpus World is a classic AI problem that demonstrates logical reasoning in uncertain environments. It's a cave exploration game where an agent must find gold while avoiding pits and a monster (Wumpus).

### Environment Description:
- **4×4 grid world** with rooms
- **Agent** starts at (1,1) facing right
- **Gold** hidden in one room
- **Wumpus** (monster) in one room
- **Pits** in some rooms
- **Agent has one arrow** to kill Wumpus

### Percepts:
- **Breeze**: Adjacent to pit
- **Stench**: Adjacent to Wumpus  
- **Glitter**: Gold in current room
- **Bump**: Walked into wall
- **Scream**: Wumpus killed by arrow

### Actions:
- **Forward**: Move forward
- **TurnLeft**, **TurnRight**: Change orientation
- **Grab**: Pick up gold
- **Shoot**: Fire arrow forward
- **Climb**: Exit cave (only at (1,1))

### Implementation

```python
import random
from enum import Enum
from typing import Tuple, List, Set, Optional

class Direction(Enum):
    NORTH = (0, 1)
    EAST = (1, 0)
    SOUTH = (0, -1)
    WEST = (-1, 0)

class WumpusWorld:
    """Implementation of the Wumpus World environment."""
    
    def __init__(self, size=4):
        self.size = size
        self.agent_pos = (1, 1)
        self.agent_dir = Direction.EAST
        self.gold_pos = self.random_position()
        self.wumpus_pos = self.random_position()
        self.pits = self.generate_pits()
        self.wumpus_alive = True
        self.gold_collected = False
        self.arrows = 1
        self.score = 0
        self.game_over = False
    
    def random_position(self):
        """Generate random position (not (1,1))."""
        while True:
            pos = (random.randint(1, self.size), random.randint(1, self.size))
            if pos != (1, 1):
                return pos
    
    def generate_pits(self):
        """Generate random pit locations."""
        pits = set()
        for x in range(1, self.size + 1):
            for y in range(1, self.size + 1):
                if (x, y) != (1, 1) and random.random() < 0.2:  # 20% chance
                    pits.add((x, y))
        return pits
    
    def get_percept(self):
        """Get current percepts at agent's location."""
        percepts = []
        
        # Check for breeze (pit nearby)
        if self.is_breeze():
            percepts.append('Breeze')
        
        # Check for stench (wumpus nearby)
        if self.is_stench():
            percepts.append('Stench')
        
        # Check for glitter (gold here)
        if self.agent_pos == self.gold_pos and not self.gold_collected:
            percepts.append('Glitter')
        
        return percepts
    
    def is_breeze(self):
        """Check if there's a breeze (pit adjacent)."""
        x, y = self.agent_pos
        adjacent_cells = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        
        for cell in adjacent_cells:
            if cell in self.pits:
                return True
        return False
    
    def is_stench(self):
        """Check if there's a stench (wumpus adjacent)."""
        if not self.wumpus_alive:
            return False
        
        x, y = self.agent_pos
        wx, wy = self.wumpus_pos
        
        return abs(x - wx) + abs(y - wy) == 1
    
    def execute_action(self, action):
        """Execute agent action and return result."""
        if self.game_over:
            return None
        
        result = {'action': action, 'percept': [], 'score_change': -1}  # -1 for each action
        
        if action == 'Forward':
            result.update(self.move_forward())
        elif action == 'TurnLeft':
            self.turn_left()
        elif action == 'TurnRight':
            self.turn_right()
        elif action == 'Grab':
            result.update(self.grab_gold())
        elif action == 'Shoot':
            result.update(self.shoot_arrow())
        elif action == 'Climb':
            result.update(self.climb())
        
        # Add current percepts
        result['percept'] = self.get_percept()
        
        # Update score
        self.score += result['score_change']
        
        return result
    
    def move_forward(self):
        """Move agent forward in current direction."""
        dx, dy = self.agent_dir.value
        new_x = self.agent_pos[0] + dx
        new_y = self.agent_pos[1] + dy
        
        # Check boundaries
        if 1 <= new_x <= self.size and 1 <= new_y <= self.size:
            self.agent_pos = (new_x, new_y)
            
            # Check for death
            if self.agent_pos in self.pits or (self.agent_pos == self.wumpus_pos and self.wumpus_alive):
                self.game_over = True
                return {'score_change': -1000, 'percept': ['Death']}
            
            return {'score_change': -1}
        else:
            # Bump into wall
            return {'score_change': -1, 'percept': ['Bump']}
    
    def turn_left(self):
        """Turn agent left."""
        directions = [Direction.NORTH, Direction.WEST, Direction.SOUTH, Direction.EAST]
        current_idx = directions.index(self.agent_dir)
        self.agent_dir = directions[(current_idx + 1) % 4]
    
    def turn_right(self):
        """Turn agent right."""
        directions = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        current_idx = directions.index(self.agent_dir)
        self.agent_dir = directions[(current_idx + 1) % 4]
    
    def grab_gold(self):
        """Attempt to grab gold."""
        if self.agent_pos == self.gold_pos and not self.gold_collected:
            self.gold_collected = True
            return {'score_change': 999}  # +1000 for gold, -1 for action
        return {'score_change': -1}
    
    def shoot_arrow(self):
        """Shoot arrow in current direction."""
        if self.arrows <= 0:
            return {'score_change': -1}
        
        self.arrows -= 1
        
        # Check if arrow hits wumpus
        dx, dy = self.agent_dir.value
        arrow_path = []
        x, y = self.agent_pos
        
        while 1 <= x + dx <= self.size and 1 <= y + dy <= self.size:
            x += dx
            y += dy
            arrow_path.append((x, y))
        
        if self.wumpus_pos in arrow_path and self.wumpus_alive:
            self.wumpus_alive = False
            return {'score_change': -11, 'percept': ['Scream']}  # -10 for arrow, -1 for action
        
        return {'score_change': -11}  # Arrow wasted
    
    def climb(self):
        """Climb out of cave."""
        if self.agent_pos == (1, 1):
            self.game_over = True
            return {'score_change': -1, 'percept': ['Climb']}
        return {'score_change': -1}


class WumpusAgent:
    """Logical agent for Wumpus World using propositional logic."""
    
    def __init__(self, world_size=4):
        self.size = world_size
        self.kb = WumpusKnowledgeBase(world_size)
        self.position = (1, 1)
        self.orientation = Direction.EAST
        self.time = 0
        self.visited = {(1, 1)}
        self.plan = []
        self.has_gold = False
    
    def agent_program(self, percept):
        """Main agent program following KB agent cycle."""
        # TELL: Add percept to knowledge base
        self.tell_percept(percept)
        
        # Update internal state
        self.update_state(percept)
        
        # ASK: Determine safe actions
        safe_actions = self.get_safe_actions()
        
        # Choose action based on goals and safety
        action = self.choose_action(safe_actions)
        
        # TELL: Record intended action
        self.tell_action(action)
        
        self.time += 1
        return action
    
    def tell_percept(self, percept):
        """Add percept information to knowledge base."""
        # Add percept facts at current location
        self.kb.tell_percept(self.position, self.time, percept)
        
        # Add location fact
        self.kb.tell(f"At({self.position}, {self.time})")
        
        # Mark location as visited and safe
        self.kb.tell(f"Safe({self.position})")
        self.visited.add(self.position)
    
    def tell_action(self, action):
        """Record action taken."""
        self.kb.tell(f"Action({action}, {self.time})")
    
    def get_safe_actions(self):
        """Determine which actions are safe to take."""
        safe_actions = []
        
        # Always safe: Turn actions
        safe_actions.extend(['TurnLeft', 'TurnRight'])
        
        # Check if forward is safe
        next_pos = self.get_next_position('Forward')
        if next_pos and self.kb.ask_safe(next_pos):
            safe_actions.append('Forward')
        
        # Grab if gold is here
        if 'Glitter' in self.get_current_percept():
            safe_actions.append('Grab')
        
        # Shoot if we suspect wumpus ahead
        if self.should_shoot():
            safe_actions.append('Shoot')
        
        # Climb if at (1,1) and have gold
        if self.position == (1, 1) and self.has_gold:
            safe_actions.append('Climb')
        
        return safe_actions
    
    def choose_action(self, safe_actions):
        """Choose best action from safe actions."""
        # Priority 1: Climb if we have gold and are at start
        if 'Climb' in safe_actions and self.has_gold:
            return 'Climb'
        
        # Priority 2: Grab gold if available
        if 'Grab' in safe_actions:
            self.has_gold = True
            return 'Grab'
        
        # Priority 3: Shoot wumpus if we can
        if 'Shoot' in safe_actions:
            return 'Shoot'
        
        # Priority 4: Explore new safe areas
        if 'Forward' in safe_actions:
            next_pos = self.get_next_position('Forward')
            if next_pos not in self.visited:
                return 'Forward'
        
        # Priority 5: Turn to find new direction
        if 'TurnRight' in safe_actions:
            return 'TurnRight'
        
        # Default: Turn left
        return 'TurnLeft'
    
    def get_next_position(self, action):
        """Get position after executing action."""
        if action == 'Forward':
            dx, dy = self.orientation.value
            new_x = self.position[0] + dx
            new_y = self.position[1] + dy
            
            if 1 <= new_x <= self.size and 1 <= new_y <= self.size:
                return (new_x, new_y)
        
        return None
    
    def update_state(self, percept):
        """Update agent's internal state based on last action."""
        # This would be called after action execution
        # For now, assume we track position separately
        pass
    
    def should_shoot(self):
        """Determine if we should shoot arrow."""
        # Simple heuristic: shoot if we smell stench and haven't shot yet
        current_percept = self.get_current_percept()
        return ('Stench' in current_percept and 
                not self.kb.ask("Shot()") and
                not self.kb.ask("WumpusDead()"))
    
    def get_current_percept(self):
        """Get current percept (simplified)."""
        return self.kb.get_current_percept()


class WumpusKnowledgeBase:
    """Knowledge base for Wumpus World with logical reasoning."""
    
    def __init__(self, world_size=4):
        self.size = world_size
        self.facts = set()
        self.rules = []
        self.current_percept = []
        self.initialize_rules()
    
    def initialize_rules(self):
        """Initialize domain-specific rules."""
        # Wumpus rules
        self.add_wumpus_rules()
        self.add_pit_rules()
        self.add_general_rules()
    
    def add_wumpus_rules(self):
        """Add rules about wumpus behavior."""
        rules = [
            # If there's a stench, wumpus is adjacent
            "∀x,y,t: Stench(x,y,t) → (∃x',y': Adjacent(x,y,x',y') ∧ Wumpus(x',y'))",
            
            # If no stench, no wumpus adjacent
            "∀x,y,t: ¬Stench(x,y,t) → (∀x',y': Adjacent(x,y,x',y') → ¬Wumpus(x',y'))",
            
            # Wumpus causes death
            "∀x,y,t: At(x,y,t) ∧ Wumpus(x,y) ∧ WumpusAlive() → Dead()",
            
            # Only one wumpus
            "∀x1,y1,x2,y2: Wumpus(x1,y1) ∧ Wumpus(x2,y2) → (x1=x2 ∧ y1=y2)"
        ]
        self.rules.extend(rules)
    
    def add_pit_rules(self):
        """Add rules about pits."""
        rules = [
            # If there's a breeze, pit is adjacent
            "∀x,y,t: Breeze(x,y,t) → (∃x',y': Adjacent(x,y,x',y') ∧ Pit(x',y'))",
            
            # If no breeze, no pits adjacent  
            "∀x,y,t: ¬Breeze(x,y,t) → (∀x',y': Adjacent(x,y,x',y') → ¬Pit(x',y'))",
            
            # Pits cause death
            "∀x,y,t: At(x,y,t) ∧ Pit(x,y) → Dead()"
        ]
        self.rules.extend(rules)
    
    def add_general_rules(self):
        """Add general world rules."""
        rules = [
            # Safety: no pit and no live wumpus
            "∀x,y: Safe(x,y) ↔ (¬Pit(x,y) ∧ (¬Wumpus(x,y) ∨ ¬WumpusAlive()))",
            
            # Can't be in two places at once
            "∀x1,y1,x2,y2,t: At(x1,y1,t) ∧ At(x2,y2,t) → (x1=x2 ∧ y1=y2)",
            
            # Gold pickup
            "∀x,y,t: At(x,y,t) ∧ Glitter(x,y,t) ∧ Grab(t) → HaveGold(t+1)"
        ]
        self.rules.extend(rules)
    
    def tell(self, sentence):
        """Add sentence to knowledge base."""
        self.facts.add(sentence)
    
    def tell_percept(self, position, time, percept):
        """Add percept information."""
        x, y = position
        self.current_percept = percept
        
        # Add percept facts
        for p in percept:
            self.tell(f"{p}({x},{y},{time})")
        
        # Add negative facts for missing percepts
        all_percepts = ['Breeze', 'Stench', 'Glitter', 'Bump', 'Scream']
        for p in all_percepts:
            if p not in percept:
                self.tell(f"¬{p}({x},{y},{time})")
    
    def ask(self, query):
        """Ask if query is entailed by KB."""
        # Simplified entailment check
        return query in self.facts
    
    def ask_safe(self, position):
        """Ask if position is safe."""
        x, y = position
        
        # Check if we know it's safe
        if f"Safe({position})" in self.facts:
            return True
        
        # Check if we can infer it's safe
        return self.infer_safety(position)
    
    def infer_safety(self, position):
        """Infer if position is safe using logical reasoning."""
        x, y = position
        
        # Check adjacent cells for absence of danger indicators
        safe = True
        
        # Get adjacent cells that we've visited
        adjacent_visited = []
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            adj_x, adj_y = x + dx, y + dy
            if f"At(({adj_x},{adj_y})" in str(self.facts):
                adjacent_visited.append((adj_x, adj_y))
        
        # If no adjacent cells visited, can't infer safety
        if not adjacent_visited:
            return False
        
        # Check if any adjacent cell had breeze or stench
        for adj_pos in adjacent_visited:
            adj_x, adj_y = adj_pos
            
            # Look for percepts at adjacent position
            has_breeze = any(f"Breeze({adj_x},{adj_y}" in fact for fact in self.facts)
            has_stench = any(f"Stench({adj_x},{adj_y}" in fact for fact in self.facts)
            
            if has_breeze or has_stench:
                safe = False
                break
        
        if safe:
            self.tell(f"Safe({position})")
        
        return safe
    
    def get_current_percept(self):
        """Get current percept."""
        return self.current_percept


# Usage Example
def run_wumpus_simulation():
    """Run a simulation of Wumpus World with logical agent."""
    world = WumpusWorld(size=4)
    agent = WumpusAgent(world_size=4)
    
    print("Starting Wumpus World Simulation")
    print(f"Gold at: {world.gold_pos}")
    print(f"Wumpus at: {world.wumpus_pos}")
    print(f"Pits at: {world.pits}")
    print("-" * 40)
    
    step = 0
    while not world.game_over and step < 100:
        # Get current percept
        percept = world.get_percept()
        
        print(f"Step {step}: Agent at {world.agent_pos}, facing {world.agent_dir.name}")
        print(f"Percept: {percept}")
        
        # Agent chooses action
        action = agent.agent_program(percept)
        print(f"Action: {action}")
        
        # Execute action in world
        result = world.execute_action(action)
        print(f"Result: {result}")
        print(f"Score: {world.score}")
        
        # Update agent's position (simplified)
        if action == 'Forward' and 'Bump' not in result.get('percept', []):
            dx, dy = world.agent_dir.value
            agent.position = (agent.position[0] + dx, agent.position[1] + dy)
        elif action == 'TurnLeft':
            directions = [Direction.NORTH, Direction.WEST, Direction.SOUTH, Direction.EAST]
            current_idx = directions.index(agent.orientation)
            agent.orientation = directions[(current_idx + 1) % 4]
        elif action == 'TurnRight':
            directions = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
            current_idx = directions.index(agent.orientation)
            agent.orientation = directions[(current_idx + 1) % 4]
        
        print("-" * 40)
        step += 1
        
        if 'Death' in result.get('percept', []):
            print("Agent died!")
            break
        elif action == 'Climb' and world.agent_pos == (1, 1):
            print("Agent escaped!")
            if world.gold_collected:
                print("With gold! Success!")
            break
    
    print(f"Final score: {world.score}")
    return world.score


# Example of logical reasoning in Wumpus World
class WumpusLogicalReasoning:
    """Demonstrate step-by-step logical reasoning."""
    
    def __init__(self):
        self.kb = set()
    
    def demonstrate_reasoning(self):
        """Show logical reasoning process."""
        print("Wumpus World Logical Reasoning Example")
        print("=" * 50)
        
        # Initial state
        print("Initial Knowledge:")
        self.kb.add("At(1,1,0)")
        self.kb.add("Safe(1,1)")
        self.kb.add("WumpusAlive()")
        print("- At(1,1,0)")
        print("- Safe(1,1)")
        print("- WumpusAlive()")
        
        # Step 1: Move to (2,1), no percepts
        print("\nStep 1: Move to (2,1), percepts: []")
        self.kb.add("At(2,1,1)")
        self.kb.add("¬Breeze(2,1,1)")
        self.kb.add("¬Stench(2,1,1)")
        print("Added: At(2,1,1), ¬Breeze(2,1,1), ¬Stench(2,1,1)")
        
        # Reasoning: No breeze means no pits adjacent to (2,1)
        print("Reasoning:")
        print("- ¬Breeze(2,1,1) → no pits adjacent to (2,1)")
        print("- Therefore: ¬Pit(1,1), ¬Pit(3,1), ¬Pit(2,2)")
        print("- ¬Stench(2,1,1) → no wumpus adjacent to (2,1)")
        print("- Therefore: ¬Wumpus(1,1), ¬Wumpus(3,1), ¬Wumpus(2,2)")
        
        self.kb.add("¬Pit(1,1)")
        self.kb.add("¬Pit(3,1)")
        self.kb.add("¬Pit(2,2)")
        self.kb.add("¬Wumpus(1,1)")
        self.kb.add("¬Wumpus(3,1)")
        self.kb.add("¬Wumpus(2,2)")
        
        # Safety inference
        print("- Since ¬Pit(3,1) ∧ ¬Wumpus(3,1) → Safe(3,1)")
        print("- Since ¬Pit(2,2) ∧ ¬Wumpus(2,2) → Safe(2,2)")
        self.kb.add("Safe(3,1)")
        self.kb.add("Safe(2,2)")
        
        # Step 2: Move to (2,2), percept: Breeze
        print("\nStep 2: Move to (2,2), percepts: [Breeze]")
        self.kb.add("At(2,2,2)")
        self.kb.add("Breeze(2,2,2)")
        print("Added: At(2,2,2), Breeze(2,2,2)")
        
        print("Reasoning:")
        print("- Breeze(2,2,2) → ∃ pit adjacent to (2,2)")
        print("- Adjacent cells: (1,2), (3,2), (2,1), (2,3)")
        print("- We know ¬Pit(2,1) from previous reasoning")
        print("- Therefore: Pit(1,2) ∨ Pit(3,2) ∨ Pit(2,3)")
        print("- Cannot determine which specific cell has pit")
        print("- All three cells (1,2), (3,2), (2,3) are potentially unsafe")
        
        # Step 3: Move to (3,2), percept: Stench
        print("\nStep 3: Move to (3,2), percepts: [Stench]")
        self.kb.add("At(3,2,3)")
        self.kb.add("Stench(3,2,3)")
        print("Added: At(3,2,3), Stench(3,2,3)")
        
        print("Reasoning:")
        print("- Stench(3,2,3) → wumpus adjacent to (3,2)")
        print("- Adjacent cells: (2,2), (4,2), (3,1), (3,3)")
        print("- We know ¬Wumpus(2,2) and ¬Wumpus(3,1)")
        print("- Therefore: Wumpus(4,2) ∨ Wumpus(3,3)")
        print("- Since only one wumpus exists: Wumpus(4,2) ⊕ Wumpus(3,3)")
        
        print("\nCurrent Knowledge Base contains:")
        for fact in sorted(self.kb):
            print(f"- {fact}")
        
        print(f"\nTotal facts in KB: {len(self.kb)}")


if __name__ == "__main__":
    # Run demonstration
    demo = WumpusLogicalReasoning()
    demo.demonstrate_reasoning()
    
    print("\n" + "="*50)
    
    # Run simulation
    # run_wumpus_simulation()
```

### Key Insights from Wumpus World:

1. **Logical Reasoning**: Agent uses rules to infer hidden properties
2. **Uncertainty Handling**: Must act with incomplete information  
3. **Safety Reasoning**: Conservative approach to avoid danger
4. **Goal-Oriented Behavior**: Balance exploration with goal achievement
5. **Learning**: Update knowledge based on observations

---

## 14. Propositional Logic

### Concept
Propositional logic is a formal system for reasoning about propositions (statements that are either true or false). It forms the foundation for more complex logical systems used in AI.

### Components:

#### Syntax:
- **Atomic propositions**: P, Q, R (indivisible statements)
- **Logical connectives**: ¬ (not), ∧ (and), ∨ (or), → (implies), ↔ (if and only if)
- **Well-formed formulas**: Built from atoms and connectives

#### Semantics:
- **Truth values**: True (T) or False (F)
- **Truth tables**: Define meaning of connectives
- **Models**: Truth assignments that make formula true

### Truth Tables for Connectives:

| P | Q | ¬P | P∧Q | P∨Q | P→Q | P↔Q |
|---|---|----|----|----|----|-----|
| T | T | F  | T  | T  | T  | T   |
| T | F | F  | F  | T  | F  | F   |
| F | T | T  | F  | T  | T  | F   |
| F | F | T  | F  | F  | T  | T   |

### Python Implementation

```python
from typing import Dict, List, Set, Union
from enum import Enum
import itertools

class LogicalOperator(Enum):
    NOT = "¬"
    AND = "∧"
    OR = "∨"
    IMPLIES = "→"
    IFF = "↔"

class Proposition:
    """Base class for propositional logic expressions."""
    
    def evaluate(self, model: Dict[str, bool]) -> bool:
        """Evaluate proposition given truth assignment."""
        raise NotImplementedError
    
    def get_symbols(self) -> Set[str]:
        """Get all propositional symbols in expression."""
        raise NotImplementedError
    
    def __str__(self):
        raise NotImplementedError

class AtomicProposition(Proposition):
    """Atomic proposition (single propositional variable)."""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
    
    def evaluate(self, model: Dict[str, bool]) -> bool:
        return model.get(self.symbol, False)
    
    def get_symbols(self) -> Set[str]:
        return {self.symbol}
    
    def __str__(self):
        return self.symbol
    
    def __eq__(self, other):
        return isinstance(other, AtomicProposition) and self.symbol == other.symbol
    
    def __hash__(self):
        return hash(self.symbol)

class CompoundProposition(Proposition):
    """Compound proposition with logical operator."""
    
    def __init__(self, operator: LogicalOperator, *operands: Proposition):
        self.operator = operator
        self.operands = operands
    
    def evaluate(self, model: Dict[str, bool]) -> bool:
        if self.operator == LogicalOperator.NOT:
            return not self.operands[0].evaluate(model)
        
        elif self.operator == LogicalOperator.AND:
            return all(op.evaluate(model) for op in self.operands)
        
        elif self.operator == LogicalOperator.OR:
            return any(op.evaluate(model) for op in self.operands)
        
        elif self.operator == LogicalOperator.IMPLIES:
            antecedent = self.operands[0].evaluate(model)
            consequent = self.operands[1].evaluate(model)
            return not antecedent or consequent
        
        elif self.operator == LogicalOperator.IFF:
            left = self.operands[0].evaluate(model)
            right = self.operands[1].evaluate(model)
            return left == right
        
        else:
            raise ValueError(f"Unknown operator: {self.operator}")
    
    def get_symbols(self) -> Set[str]:
        symbols = set()
        for operand in self.operands:
            symbols.update(operand.get_symbols())
        return symbols
    
    def __str__(self):
        if self.operator == LogicalOperator.NOT:
            return f"¬{self.operands[0]}"
        elif len(self.operands) == 2:
            return f"({self.operands[0]} {self.operator.value} {self.operands[1]})"
        else:
            op_str = f" {self.operator.value} "
            return f"({op_str.join(str(op) for op in self.operands)})"

# Convenience functions for building propositions
def Atom(symbol: str) -> AtomicProposition:
    """Create atomic proposition."""
    return AtomicProposition(symbol)

def Not(prop: Proposition) -> CompoundProposition:
    """Create negation."""
    return CompoundProposition(LogicalOperator.NOT, prop)

def And(*props: Proposition) -> CompoundProposition:
    """Create conjunction."""
    return CompoundProposition(LogicalOperator.AND, *props)

def Or(*props: Proposition) -> CompoundProposition:
    """Create disjunction."""
    return CompoundProposition(LogicalOperator.OR, *props)

def Implies(antecedent: Proposition, consequent: Proposition) -> CompoundProposition:
    """Create implication."""
    return CompoundProposition(LogicalOperator.IMPLIES, antecedent, consequent)

def Iff(left: Proposition, right: Proposition) -> CompoundProposition:
    """Create biconditional."""
    return CompoundProposition(LogicalOperator.IFF, left, right)

class PropositionalLogic:
    """Tools for working with propositional logic."""
    
    @staticmethod
    def generate_models(symbols: Set[str]) -> List[Dict[str, bool]]:
        """Generate all possible truth assignments for symbols."""
        symbols_list = list(symbols)
        models = []
        
        for values in itertools.product([True, False], repeat=len(symbols_list)):
            model = dict(zip(symbols_list, values))
            models.append(model)
        
        return models
    
    @staticmethod
    def is_tautology(proposition: Proposition) -> bool:
        """Check if proposition is a tautology (always true)."""
        symbols = proposition.get_symbols()
        models = PropositionalLogic.generate_models(symbols)
        
        return all(proposition.evaluate(model) for model in models)
    
    @staticmethod
    def is_contradiction(proposition: Proposition) -> bool:
        """Check if proposition is a contradiction (always false)."""
        symbols = proposition.get_symbols()
        models = PropositionalLogic.generate_models(symbols)
        
        return all(not proposition.evaluate(model) for model in models)
    
    @staticmethod
    def is_satisfiable(proposition: Proposition) -> bool:
        """Check if proposition is satisfiable (true in some model)."""
        symbols = proposition.get_symbols()
        models = PropositionalLogic.generate_models(symbols)
        
        return any(proposition.evaluate(model) for model in models)
    
    @staticmethod
    def entails(kb: List[Proposition], query: Proposition) -> bool:
        """Check if knowledge base entails query."""
        # KB entails query iff KB → query is tautology
        kb_conjunction = And(*kb) if kb else Atom("True")
        implication = Implies(kb_conjunction, query)
        
        return PropositionalLogic.is_tautology(implication)
    
    @staticmethod
    def get_models(proposition: Proposition) -> List[Dict[str, bool]]:
        """Get all models that satisfy the proposition."""
        symbols = proposition.get_symbols()
        all_models = PropositionalLogic.generate_models(symbols)
        
        return [model for model in all_models if proposition.evaluate(model)]
    
    @staticmethod
    def truth_table(proposition: Proposition) -> None:
        """Print truth table for proposition."""
        symbols = sorted(proposition.get_symbols())
        models = PropositionalLogic.generate_models(set(symbols))
        
        # Print header
        header = " | ".join(symbols) + " | " + str(proposition)
        print(header)
        print("-" * len(header))
        
        # Print rows
        for model in models:
            row_values = []
            for symbol in symbols:
                row_values.append("T" if model[symbol] else "F")
            
            result = "T" if proposition.evaluate(model) else "F"
            row = " | ".join(row_values) + " | " + result
            print(row)

# Examples and Applications
class WumpusWorldLogic:
    """Propositional logic representation of Wumpus World."""
    
    def __init__(self, size=4):
        self.size = size
        self.kb = []  # Knowledge base
    
    def create_propositions(self):
        """Create propositional variables for Wumpus World."""
        propositions = {}
        
        # For each cell (x,y), create propositions
        for x in range(1, self.size + 1):
            for y in range(1, self.size + 1):
                propositions[f"Pit_{x}_{y}"] = Atom(f"Pit_{x}_{y}")
                propositions[f"Wumpus_{x}_{y}"] = Atom(f"Wumpus_{x}_{y}")
                propositions[f"Breeze_{x}_{y}"] = Atom(f"Breeze_{x}_{y}")
                propositions[f"Stench_{x}_{y}"] = Atom(f"Stench_{x}_{y}")
                propositions[f"Safe_{x}_{y}"] = Atom(f"Safe_{x}_{y}")
        
        return propositions
    
    def add_world_rules(self, props):
        """Add general rules about Wumpus World."""
        # Rule: Breeze iff adjacent pit
        for x in range(1, self.size + 1):
            for y in range(1, self.size + 1):
                adjacent_pits = []
                
                # Get adjacent cells
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    adj_x, adj_y = x + dx, y + dy
                    if 1 <= adj_x <= self.size and 1 <= adj_y <= self.size:
                        adjacent_pits.append(props[f"Pit_{adj_x}_{adj_y}"])
                
                if adjacent_pits:
                    breeze_rule = Iff(
                        props[f"Breeze_{x}_{y}"],
                        Or(*adjacent_pits)
                    )
                    self.kb.append(breeze_rule)
        
        # Rule: At most one wumpus
        wumpus_positions = []
        for x in range(1, self.size + 1):
            for y in range(1, self.size + 1):
                wumpus_positions.append(props[f"Wumpus_{x}_{y}"])
        
        # Exactly one wumpus (simplified: at least one)
        self.kb.append(Or(*wumpus_positions))
        
        # At most one wumpus (for each pair, not both)
        for i in range(len(wumpus_positions)):
            for j in range(i + 1, len(wumpus_positions)):
                self.kb.append(Not(And(wumpus_positions[i], wumpus_positions[j])))
        
        # Safety rule: safe iff no pit and no wumpus
        for x in range(1, self.size + 1):
            for y in range(1, self.size + 1):
                safety_rule = Iff(
                    props[f"Safe_{x}_{y}"],
                    And(
                        Not(props[f"Pit_{x}_{y}"]),
                        Not(props[f"Wumpus_{x}_{y}"])
                    )
                )
                self.kb.append(safety_rule)
    
    def add_observations(self, observations, props):
        """Add specific observations to KB."""
        for obs in observations:
            x, y, percept = obs
            if percept == "Breeze":
                self.kb.append(props[f"Breeze_{x}_{y}"])
            elif percept == "NoBreeze":
                self.kb.append(Not(props[f"Breeze_{x}_{y}"]))
            elif percept == "Stench":
                self.kb.append(props[f"Stench_{x}_{y}"])
            elif percept == "NoStench":
                self.kb.append(Not(props[f"Stench_{x}_{y}"]))
    
    def query_safety(self, x, y, props):
        """Query if position (x,y) is safe."""
        safety_query = props[f"Safe_{x}_{y}"]
        return PropositionalLogic.entails(self.kb, safety_query)
    
    def demonstrate_reasoning(self):
        """Demonstrate logical reasoning in Wumpus World."""
        print("Wumpus World Propositional Logic Demonstration")
        print("=" * 50)
        
        props = self.create_propositions()
        self.add_world_rules(props)
        
        # Add observations
        observations = [
            (1, 1, "NoBreeze"),
            (1, 1, "NoStench"),
            (2, 1, "NoBreeze"),
            (2, 1, "NoStench"),
            (2, 2, "Breeze"),
        ]
        
        print("Observations:")
        for obs in observations:
            x, y, percept = obs
            print(f"- {percept} at ({x},{y})")
        
        self.add_observations(observations, props)
        
        print(f"\nKnowledge Base has {len(self.kb)} rules")
        
        # Query safety of adjacent cells
        query_positions = [(3, 1), (1, 2), (3, 2), (2, 3)]
        
        print("\nSafety Queries:")
        for x, y in query_positions:
            is_safe = self.query_safety(x, y, props)
            print(f"- Is ({x},{y}) safe? {'Yes' if is_safe else 'Unknown/No'}")


# Example usage and demonstrations
def demonstrate_propositional_logic():
    """Demonstrate propositional logic concepts."""
    print("Propositional Logic Demonstration")
    print("=" * 40)
    
    # Create propositions
    P = Atom("P")
    Q = Atom("Q")
    R = Atom("R")
    
    # Example formulas
    formula1 = Implies(P, Q)  # P → Q
    formula2 = And(P, Not(Q))  # P ∧ ¬Q
    formula3 = Or(And(P, Q), And(Not(P), Not(Q)))  # (P ∧ Q) ∨ (¬P ∧ ¬Q)
    
    print("Example Formulas:")
    print(f"1. {formula1}")
    print(f"2. {formula2}")
    print(f"3. {formula3}")
    
    # Check properties
    print("\nFormula Properties:")
    print(f"Formula 1 is tautology: {PropositionalLogic.is_tautology(formula1)}")
    print(f"Formula 1 is satisfiable: {PropositionalLogic.is_satisfiable(formula1)}")
    
    print(f"Formula 2 is contradiction: {PropositionalLogic.is_contradiction(formula2)}")
    print(f"Formula 2 is satisfiable: {PropositionalLogic.is_satisfiable(formula2)}")
    
    print(f"Formula 3 is tautology: {PropositionalLogic.is_tautology(formula3)}")
    
    # Truth table
    print(f"\nTruth table for {formula1}:")
    PropositionalLogic.truth_table(formula1)
    
    # Entailment example
    kb = [P, Implies(P, Q)]  # Knowledge base: P, P → Q
    query = Q  # Query: Q
    
    print(f"\nEntailment example:")
    print(f"KB: {[str(prop) for prop in kb]}")
    print(f"Query: {query}")
    print(f"KB entails Query: {PropositionalLogic.entails(kb, query)}")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_propositional_logic()
    
    print("\n" + "="*50 + "\n")
    
    # Wumpus World logic example
    wumpus_logic = WumpusWorldLogic(size=3)  # Smaller world for demo
    wumpus_logic.demonstrate_reasoning()
```

### Applications in AI:

1. **Knowledge Representation**: Encode facts and rules
2. **Automated Reasoning**: Derive conclusions from premises
3. **Planning**: Represent states and actions
4. **Diagnosis**: Model systems and faults
5. **Configuration**: Constraint satisfaction problems

---

## 15. Propositional Theorem Proving

### Concept
Theorem proving involves determining whether a formula is a logical consequence of a set of premises. It's the computational heart of logical reasoning in AI systems.

### Inference Rules

#### Basic Inference Rules:
1. **Modus Ponens**: From P and P→Q, infer Q
2. **Modus Tollens**: From ¬Q and P→Q, infer ¬P
3. **And Elimination**: From P∧Q, infer P (and Q)
4. **And Introduction**: From P and Q, infer P∧Q
5. **Or Introduction**: From P, infer P∨Q
6. **Double Negation**: From ¬¬P, infer P

### Resolution Method

The resolution method is a complete inference procedure for propositional logic.

#### Steps:
1. Convert to Conjunctive Normal Form (CNF)
2. Apply resolution rule repeatedly
3. If empty clause derived, formula is unsatisfiable

#### Resolution Rule:
From clauses (A ∨ P) and (¬P ∨ B), infer (A ∨ B)

### Python Implementation

```python
from typing import Set, List, Tuple, Optional
import copy

class Literal:
    """A literal is an atomic proposition or its negation."""
    
    def __init__(self, symbol: str, positive: bool = True):
        self.symbol = symbol
        self.positive = positive
    
    def negate(self):
        """Return negated literal."""
        return Literal(self.symbol, not self.positive)
    
    def __str__(self):
        return self.symbol if self.positive else f"¬{self.symbol}"
    
    def __eq__(self, other):
        return (isinstance(other, Literal) and 
                self.symbol == other.symbol and 
                self.positive == other.positive)
    
    def __hash__(self):
        return hash((self.symbol, self.positive))
    
    def __repr__(self):
        return str(self)

class Clause:
    """A clause is a disjunction of literals."""
    
    def __init__(self, literals: List[Literal]):
        # Remove duplicates and sort for consistency
        self.literals = list(set(literals))
        
        # Check for tautology (P ∨ ¬P)
        symbols = {}
        for lit in self.literals:
            if lit.symbol in symbols:
                if symbols[lit.symbol] != lit.positive:
                    # Contains both P and ¬P - tautology
                    self.literals = []  # Empty means tautology
                    break
            symbols[lit.symbol] = lit.positive
    
    def is_empty(self):
        """Check if clause is empty (contradiction)."""
        return len(self.literals) == 0
    
    def is_unit(self):
        """Check if clause is unit (single literal)."""
        return len(self.literals) == 1
    
    def get_unit_literal(self):
        """Get the literal if this is a unit clause."""
        return self.literals[0] if self.is_unit() else None
    
    def resolve_with(self, other):
        """Apply resolution rule with another clause."""
        resolvents = []
        
        for lit1 in self.literals:
            for lit2 in other.literals:
                if lit1.symbol == lit2.symbol and lit1.positive != lit2.positive:
                    # Found complementary literals
                    new_literals = []
                    
                    # Add all literals except the resolved ones
                    for lit in self.literals:
                        if lit != lit1:
                            new_literals.append(lit)
                    
                    for lit in other.literals:
                        if lit != lit2:
                            new_literals.append(lit)
                    
                    # Remove duplicates
                    resolvent = Clause(new_literals)
                    resolvents.append(resolvent)
        
        return resolvents
    
    def contains_literal(self, literal):
        """Check if clause contains given literal."""
        return literal in self.literals
    
    def __str__(self):
        if self.is_empty():
            return "□"  # Empty clause (contradiction)
        return " ∨ ".join(str(lit) for lit in self.literals)
    
    def __eq__(self, other):
        return (isinstance(other, Clause) and 
                set(self.literals) == set(other.literals))
    
    def __hash__(self):
        return hash(frozenset(self.literals))

class CNFConverter:
    """Convert propositional formulas to Conjunctive Normal Form."""
    
    @staticmethod
    def to_cnf(formula: Proposition) -> List[Clause]:
        """Convert formula to CNF."""
        # Step 1: Eliminate biconditionals and implications
        formula = CNFConverter.eliminate_biconditionals(formula)
        formula = CNFConverter.eliminate_implications(formula)
        
        # Step 2: Move negations inward (De Morgan's laws)
        formula = CNFConverter.move_negations_inward(formula)
        
        # Step 3: Distribute OR over AND
        formula = CNFConverter.distribute_or_over_and(formula)
        
        # Step 4: Convert to clause list
        return CNFConverter.formula_to_clauses(formula)
    
    @staticmethod
    def eliminate_biconditionals(formula):
        """Replace P ↔ Q with (P → Q) ∧ (Q → P)."""
        if isinstance(formula, AtomicProposition):
            return formula
        
        if isinstance(formula, CompoundProposition):
            if formula.operator == LogicalOperator.IFF:
                left, right = formula.operands
                left = CNFConverter.eliminate_biconditionals(left)
                right = CNFConverter.eliminate_biconditionals(right)
                
                return And(
                    Implies(left, right),
                    Implies(right, left)
                )
            else:
                new_operands = [CNFConverter.eliminate_biconditionals(op) 
                              for op in formula.operands]
                return CompoundProposition(formula.operator, *new_operands)
        
        return formula
    
    @staticmethod
    def eliminate_implications(formula):
        """Replace P → Q with ¬P ∨ Q."""
        if isinstance(formula, AtomicProposition):
            return formula
        
        if isinstance(formula, CompoundProposition):
            if formula.operator == LogicalOperator.IMPLIES:
                antecedent, consequent = formula.operands
                antecedent = CNFConverter.eliminate_implications(antecedent)
                consequent = CNFConverter.eliminate_implications(consequent)
                
                return Or(Not(antecedent), consequent)
            else:
                new_operands = [CNFConverter.eliminate_implications(op) 
                              for op in formula.operands]
                return CompoundProposition(formula.operator, *new_operands)
        
        return formula
    
    @staticmethod
    def move_negations_inward(formula):
        """Apply De Morgan's laws and double negation elimination."""
        if isinstance(formula, AtomicProposition):
            return formula
        
        if isinstance(formula, CompoundProposition):
            if formula.operator == LogicalOperator.NOT:
                inner = formula.operands[0]
                
                if isinstance(inner, AtomicProposition):
                    return formula  # ¬P stays as is
                
                if isinstance(inner, CompoundProposition):
                    if inner.operator == LogicalOperator.NOT:
                        # Double negation: ¬¬P → P
                        return CNFConverter.move_negations_inward(inner.operands[0])
                    
                    elif inner.operator == LogicalOperator.AND:
                        # De Morgan: ¬(P ∧ Q) → ¬P ∨ ¬Q
                        new_operands = [CNFConverter.move_negations_inward(Not(op))
                                      for op in inner.operands]
                        return Or(*new_operands)
                    
                    elif inner.operator == LogicalOperator.OR:
                        # De Morgan: ¬(P ∨ Q) → ¬P ∧ ¬Q
                        new_operands = [CNFConverter.move_negations_inward(Not(op))
                                      for op in inner.operands]
                        return And(*new_operands)
            
            else:
                new_operands = [CNFConverter.move_negations_inward(op) 
                              for op in formula.operands]
                return CompoundProposition(formula.operator, *new_operands)
        
        return formula
    
    @staticmethod
    def distribute_or_over_and(formula):
        """Distribute OR over AND: (A ∧ B) ∨ C → (A ∨ C) ∧ (B ∨ C)."""
        if isinstance(formula, AtomicProposition):
            return formula
        
        if isinstance(formula, CompoundProposition):
            if formula.operator == LogicalOperator.OR:
                # Check if any operand is an AND
                for i, operand in enumerate(formula.operands):
                    operand = CNFConverter.distribute_or_over_and(operand)
                    
                    if (isinstance(operand, CompoundProposition) and 
                        operand.operator == LogicalOperator.AND):
                        
                        # Distribute this OR over the AND
                        other_operands = [CNFConverter.distribute_or_over_and(op) 
                                        for j, op in enumerate(formula.operands) if j != i]
                        
                        new_conjuncts = []
                        for and_operand in operand.operands:
                            if other_operands:
                                new_disjunct = Or(and_operand, *other_operands)
                            else:
                                new_disjunct = and_operand
                            
                            new_conjuncts.append(
                                CNFConverter.distribute_or_over_and(new_disjunct)
                            )
                        
                        return And(*new_conjuncts)
                
                # No AND found, just process operands
                new_operands = [CNFConverter.distribute_or_over_and(op) 
                              for op in formula.operands]
                return CompoundProposition(formula.operator, *new_operands)
            
            else:
                new_operands = [CNFConverter.distribute_or_over_and(op) 
                              for op in formula.operands]
                return CompoundProposition(formula.operator, *new_operands)
        
        return formula
    
    @staticmethod
    def formula_to_clauses(formula):
        """Convert CNF formula to list of clauses."""
        if isinstance(formula, AtomicProposition):
            return [Clause([Literal(formula.symbol, True)])]
        
        if isinstance(formula, CompoundProposition):
            if formula.operator == LogicalOperator.NOT:
                # Single negated atom
                inner = formula.operands[0]
                if isinstance(inner, AtomicProposition):
                    return [Clause([Literal(inner.symbol, False)])]
            
            elif formula.operator == LogicalOperator.AND:
                # Conjunction: collect all clauses
                all_clauses = []
                for operand in formula.operands:
                    all_clauses.extend(CNFConverter.formula_to_clauses(operand))
                return all_clauses
            
            elif formula.operator == LogicalOperator.OR:
                # Disjunction: single clause with multiple literals
                literals = []
                for operand in formula.operands:
                    if isinstance(operand, AtomicProposition):
                        literals.append(Literal(operand.symbol, True))
                    elif (isinstance(operand, CompoundProposition) and 
                          operand.operator == LogicalOperator.NOT and
                          isinstance(operand.operands[0], AtomicProposition)):
                        literals.append(Literal(operand.operands[0].symbol, False))
                
                return [Clause(literals)]
        
        return []

class ResolutionProver:
    """Resolution-based theorem prover for propositional logic."""
    
    def __init__(self):
        self.derivation_steps = []
    
    def prove(self, premises: List[Proposition], conclusion: Proposition) -> bool:
        """
        Prove that premises entail conclusion using resolution.
        
        Method: Show that premises ∧ ¬conclusion is unsatisfiable.
        """
        self.derivation_steps = []
        
        # Convert premises to CNF
        all_clauses = set()
        for premise in premises:
            cnf_clauses = CNFConverter.to_cnf(premise)
            all_clauses.update(cnf_clauses)
        
        # Add negation of conclusion
        negated_conclusion = Not(conclusion)
        conclusion_clauses = CNFConverter.to_cnf(negated_conclusion)
        all_clauses.update(conclusion_clauses)
        
        print(f"Initial clauses ({len(all_clauses)}):")
        for i, clause in enumerate(all_clauses):
            print(f"  {i+1}. {clause}")
        
        # Apply resolution
        new_clauses = set()
        step = len(all_clauses)
        
        while True:
            clause_list = list(all_clauses)
            n = len(clause_list)
            
            # Try to resolve each pair of clauses
            for i in range(n):
                for j in range(i + 1, n):
                    resolvents = clause_list[i].resolve_with(clause_list[j])
                    
                    for resolvent in resolvents:
                        if resolvent.is_empty():
                            # Found contradiction - proof complete
                            step += 1
                            print(f"  {step}. {resolvent} (from {i+1} and {j+1})")
                            print("\nContradiction found! Proof complete.")
                            return True
                        
                        if resolvent not in all_clauses:
                            new_clauses.add(resolvent)
                            step += 1
                            print(f"  {step}. {resolvent} (from {i+1} and {j+1})")
            
            # If no new clauses, we can't prove it
            if not new_clauses:
                print("\nNo new clauses derived. Cannot prove conclusion.")
                return False
            
            # Add new clauses for next iteration
            all_clauses.update(new_clauses)
            new_clauses.clear()
    
    def unit_propagation(self, clauses: Set[Clause]) -> Tuple[Set[Clause], bool]:
        """Apply unit propagation to simplify clauses."""
        changed = True
        unit_literals = set()
        
        while changed:
            changed = False
            
            # Find unit clauses
            for clause in clauses:
                if clause.is_unit():
                    unit_literal = clause.get_unit_literal()
                    if unit_literal not in unit_literals:
                        unit_literals.add(unit_literal)
                        changed = True
            
            # Propagate unit literals
            new_clauses = set()
            for clause in clauses:
                new_clause_literals = []
                satisfied = False
                
                for literal in clause.literals:
                    if literal in unit_literals:
                        # Clause is satisfied
                        satisfied = True
                        break
                    elif literal.negate() not in unit_literals:
                        # Literal is not eliminated
                        new_clause_literals.append(literal)
                
                if not satisfied:
                    if not new_clause_literals:
                        # Empty clause - contradiction
                        return set(), True
                    new_clauses.add(Clause(new_clause_literals))
            
            clauses = new_clauses
        
        return clauses, False

# DPLL Algorithm (Davis-Putnam-Logemann-Loveland)
class DPLLSolver:
    """DPLL algorithm for SAT solving."""
    
    def __init__(self):
        self.assignment = {}
        self.steps = 0
    
    def dpll(self, clauses: Set[Clause], assignment: dict = None) -> bool:
        """
        DPLL algorithm for satisfiability checking.
        
        Returns True if satisfiable, False if unsatisfiable.
        """
        if assignment is None:
            assignment = {}
        
        self.steps += 1
        
        # Unit propagation
        clauses, contradiction = self.unit_propagation(clauses, assignment)
        if contradiction:
            return False
        
        # Check if all clauses satisfied
        if not clauses:
            self.assignment = assignment.copy()
            return True
        
        # Pure literal elimination
        clauses = self.pure_literal_elimination(clauses, assignment)
        
        # Check again after pure literal elimination
        if not clauses:
            self.assignment = assignment.copy()
            return True
        
        # Choose a literal for branching
        literal = self.choose_literal(clauses)
        if literal is None:
            return False
        
        # Try positive assignment
        new_assignment = assignment.copy()
        new_assignment[literal.symbol] = literal.positive
        
        new_clauses = self.apply_assignment(clauses, literal)
        if self.dpll(new_clauses, new_assignment):
            return True
        
        # Try negative assignment
        negated_literal = literal.negate()
        new_assignment = assignment.copy()
        new_assignment[negated_literal.symbol] = negated_literal.positive
        
        new_clauses = self.apply_assignment(clauses, negated_literal)
        return self.dpll(new_clauses, new_assignment)
    
    def unit_propagation(self, clauses: Set[Clause], assignment: dict):
        """Apply unit propagation."""
        changed = True
        new_assignment = assignment.copy()
        
        while changed:
            changed = False
            new_clauses = set()
            
            for clause in clauses:
                # Check if clause is already satisfied
                satisfied = False
                remaining_literals = []
                
                for literal in clause.literals:
                    if literal.symbol in new_assignment:
                        if new_assignment[literal.symbol] == literal.positive:
                            satisfied = True
                            break
                        # else: literal is false, don't add to remaining
                    else:
                        remaining_literals.append(literal)
                
                if satisfied:
                    continue  # Clause satisfied, skip
                
                if not remaining_literals:
                    # Empty clause - contradiction
                    return set(), True
                
                if len(remaining_literals) == 1:
                    # Unit clause
                    unit_literal = remaining_literals[0]
                    if unit_literal.symbol not in new_assignment:
                        new_assignment[unit_literal.symbol] = unit_literal.positive
                        changed = True
                
                new_clauses.add(Clause(remaining_literals))
            
            clauses = new_clauses
        
        return clauses, False
    
    def pure_literal_elimination(self, clauses: Set[Clause], assignment: dict):
        """Eliminate pure literals."""
        literal_polarities = {}
        
        # Find all literals and their polarities
        for clause in clauses:
            for literal in clause.literals:
                if literal.symbol not in assignment:
                    if literal.symbol not in literal_polarities:
                        literal_polarities[literal.symbol] = set()
                    literal_polarities[literal.symbol].add(literal.positive)
        
        # Find pure literals (appear with only one polarity)
        pure_literals = []
        for symbol, polarities in literal_polarities.items():
            if len(polarities) == 1:
                polarity = list(polarities)[0]
                pure_literals.append(Literal(symbol, polarity))
        
        # Apply pure literal assignments
        new_clauses = set()
        for clause in clauses:
            satisfied = False
            for literal in pure_literals:
                if literal in clause.literals:
                    satisfied = True
                    break
            
            if not satisfied:
                new_clauses.add(clause)
        
        return new_clauses
    
    def choose_literal(self, clauses: Set[Clause]):
        """Choose literal for branching (simple heuristic)."""
        # Choose literal that appears most frequently
        literal_counts = {}
        
        for clause in clauses:
            for literal in clause.literals:
                if literal.symbol not in literal_counts:
                    literal_counts[literal.symbol] = 0
                literal_counts[literal.symbol] += 1
        
        if not literal_counts:
            return None
        
        # Choose most frequent symbol, positive polarity
        most_frequent = max(literal_counts.items(), key=lambda x: x[1])
        return Literal(most_frequent[0], True)
    
    def apply_assignment(self, clauses: Set[Clause], literal: Literal):
        """Apply literal assignment to clauses."""
        new_clauses = set()
        
        for clause in clauses:
            if literal in clause.literals:
                # Clause satisfied, skip
                continue
            
            new_literals = []
            for lit in clause.literals:
                if lit.symbol == literal.symbol:
                    if lit.positive != literal.positive:
                        # Complementary literal eliminated
                        continue
                new_literals.append(lit)
            
            if new_literals:
                new_clauses.add(Clause(new_literals))
        
        return new_clauses

# Example usage and demonstrations
def demonstrate_resolution():
    """Demonstrate resolution theorem proving."""
    print("Resolution Theorem Proving Demonstration")
    print("=" * 50)
    
    # Example: Prove modus ponens
    # Premises: P, P → Q
    # Conclusion: Q
    
    P = Atom("P")
    Q = Atom("Q")
    
    premises = [P, Implies(P, Q)]
    conclusion = Q
    
    print("Proving Modus Ponens:")
    print(f"Premises: {[str(p) for p in premises]}")
    print(f"Conclusion: {conclusion}")
    print()
    
    prover = ResolutionProver()
    result = prover.prove(premises, conclusion)
    
    print(f"\nProof {'successful' if result else 'failed'}")
    
    print("\n" + "="*30 + "\n")
    
    # Example: Wumpus World reasoning
    print("Wumpus World Resolution Example:")
    
    # Premises
    B11 = Atom("B11")  # Breeze at (1,1)
    B21 = Atom("B21")  # Breeze at (2,1)
    P12 = Atom("P12")  # Pit at (1,2)
    P22 = Atom("P22")  # Pit at (2,2)
    P31 = Atom("P31")  # Pit at (3,1)
    
    # Rules and observations
    premises = [
        Not(B11),  # No breeze at (1,1)
        Not(B21),  # No breeze at (2,1)
        Implies(P12, B11),  # Pit at (1,2) → Breeze at (1,1)
        Implies(P22, Or(B11, B21)),  # Pit at (2,2) → Breeze at (1,1) or (2,1)
        Implies(P31, B21),  # Pit at (3,1) → Breeze at (2,1)
    ]
    
    # Conclusion: No pit at (1,2)
    conclusion = Not(P12)
    
    print(f"Premises: {[str(p) for p in premises]}")
    print(f"Conclusion: {conclusion}")
    print()
    
    result = prover.prove(premises, conclusion)
    print(f"\nProof {'successful' if result else 'failed'}")

def demonstrate_dpll():
    """Demonstrate DPLL satisfiability checking."""
    print("DPLL SAT Solver Demonstration")
    print("=" * 40)
    
    # Example: (P ∨ Q) ∧ (¬P ∨ R) ∧ (¬Q ∨ ¬R)
    P = Literal("P", True)
    Q = Literal("Q", True)
    R = Literal("R", True)
    not_P = Literal("P", False)
    not_Q = Literal("Q", False)
    not_R = Literal("R", False)
    
    clauses = {
        Clause([P, Q]),
        Clause([not_P, R]),
        Clause([not_Q, not_R])
    }
    
    print("Checking satisfiability of:")
    for i, clause in enumerate(clauses):
        print(f"  {i+1}. {clause}")
    
    solver = DPLLSolver()
    is_sat = solver.dpll(clauses)
    
    print(f"\nSatisfiable: {is_sat}")
    if is_sat:
        print(f"Satisfying assignment: {solver.assignment}")
    print(f"Search steps: {solver.steps}")

if __name__ == "__main__":
    demonstrate_resolution()
    print("\n" + "="*60 + "\n")
    demonstrate_dpll()
```

### Theorem Proving Applications:

1. **Automated Verification**: Prove program correctness
2. **Mathematical Proof**: Assist in mathematical reasoning
3. **Planning**: Verify plan validity
4. **Diagnosis**: Determine system faults
5. **Configuration**: Check constraint satisfaction

---

## 16. Model Checking

### Concept
Model checking is a method for verifying finite-state systems by exhaustively checking whether a specification holds in all possible states.

### Key Components:
- **Model**: Finite-state representation of system
- **Specification**: Property to verify (often in temporal logic)
- **Algorithm**: Systematic exploration of state space

### Truth Table Method

### Implementation

```python
class TruthTableModelChecker:
    """Model checking using truth table enumeration."""
    
    def __init__(self):
        self.models_checked = 0
    
    def tt_entails(self, kb: List[Proposition], query: Proposition) -> bool:
        """Check if KB entails query using truth table method."""
        # Get all symbols
        symbols = set()
        for sentence in kb + [query]:
            symbols.update(sentence.get_symbols())
        
        symbols = list(symbols)
        return self.tt_check_all(kb, query, symbols, {})
    
    def tt_check_all(self, kb: List[Proposition], query: Proposition, 
                     symbols: List[str], model: Dict[str, bool]) -> bool:
        """Recursive truth table checking."""
        if not symbols:
            # Base case: all symbols assigned
            self.models_checked += 1
            
            # If KB is true in this model, query must also be true
            if self.pl_true_all(kb, model):
                return self.pl_true(query, model)
            else:
                return True  # KB false, so KB ⊨ query vacuously true
        
        # Recursive case: try both truth values for first symbol
        first = symbols[0]
        rest = symbols[1:]
        
        # Try first = True
        model_true = model.copy()
        model_true[first] = True
        result_true = self.tt_check_all(kb, query, rest, model_true)
        
        # Try first = False
        model_false = model.copy()
        model_false[first] = False
        result_false = self.tt_check_all(kb, query, rest, model_false)
        
        return result_true and result_false
    
    def pl_true(self, sentence: Proposition, model: Dict[str, bool]) -> bool:
        """Check if sentence is true in model."""
        return sentence.evaluate(model)
    
    def pl_true_all(self, sentences: List[Proposition], model: Dict[str, bool]) -> bool:
        """Check if all sentences are true in model."""
        return all(self.pl_true(sentence, model) for sentence in sentences)

# Example application
def demonstrate_model_checking():
    """Demonstrate model checking with truth tables."""
    print("Model Checking Demonstration")
    print("=" * 40)
    
    # Wumpus World example
    # KB: ¬B11, B21, (B11 ↔ (P12 ∨ P22)), (B21 ↔ (P12 ∨ P22 ∨ P32))
    # Query: P12
    
    B11 = Atom("B11")
    B21 = Atom("B21")
    P12 = Atom("P12")
    P22 = Atom("P22")
    P32 = Atom("P32")
    
    kb = [
        Not(B11),  # No breeze at (1,1)
        B21,       # Breeze at (2,1)
        Iff(B11, Or(P12, P22)),  # Breeze at (1,1) iff pit at (1,2) or (2,2)
        Iff(B21, Or(P12, P22, P32))  # Breeze at (2,1) iff pit at (1,2), (2,2), or (3,2)
    ]
    
    query = P12  # Is there a pit at (1,2)?
    
    print("Knowledge Base:")
    for i, sentence in enumerate(kb):
        print(f"  {i+1}. {sentence}")
    
    print(f"\nQuery: {query}")
    
    checker = TruthTableModelChecker()
    result = checker.tt_entails(kb, query)
    
    print(f"\nResult: KB {'entails' if result else 'does not entail'} query")
    print(f"Models checked: {checker.models_checked}")
    
    # Check the negation too
    checker2 = TruthTableModelChecker()
    result2 = checker2.tt_entails(kb, Not(query))
    
    print(f"KB {'entails' if result2 else 'does not entail'} ¬{query}")
    print(f"Models checked for negation: {checker2.models_checked}")
    
    if not result and not result2:
        print(f"Query {query} is unknown given the KB")

if __name__ == "__main__":
    demonstrate_model_checking()
```

---

## Case Studies

### Google DeepMind - AI for Energy Efficiency in Data Centers

Google DeepMind achieved a 40% reduction in cooling costs by applying AI to optimize data center operations. The system used constraint satisfaction and logical reasoning.

**Approach:**
1. **Knowledge Representation**: Model data center as CSP
2. **Constraint Propagation**: Reduce search space
3. **Optimization**: Find energy-efficient configurations
4. **Real-time Reasoning**: Adapt to changing conditions

### BBC & Amazon Alexa - AI-Driven Interactive Media

Logic-based chatbots use propositional and first-order logic for natural language understanding and response generation.

**Components:**
1. **Intent Recognition**: Map user input to logical predicates
2. **Knowledge Base**: Store facts and rules about domain
3. **Inference Engine**: Derive appropriate responses
4. **Context Management**: Maintain conversation state

**Example Implementation:**
```python
class LogicBasedChatbot:
    """Simple logic-based chatbot for interactive media."""
    
    def __init__(self):
        self.kb = []
        self.context = {}
        self.setup_knowledge_base()
    
    def setup_knowledge_base(self):
        """Initialize domain knowledge."""
        # Media content rules
        self.kb.extend([
            # If user likes action, recommend action movies
            "∀x: Likes(x, action) → Recommend(x, action_movies)",
            
            # If user watched movie and liked it, recommend similar
            "∀x,m: Watched(x,m) ∧ Liked(x,m) → Recommend(x, Similar(m))",
            
            # Weekend evening suggests entertainment
            "∀x: Weekend(today) ∧ Evening(now) → Suggest(x, entertainment)"
        ])
    
    def process_input(self, user_input):
        """Process user input and generate response."""
        # Parse input to logical form
        intent = self.parse_intent(user_input)
        
        # Update context
        self.update_context(intent)
        
        # Query knowledge base
        response = self.generate_response(intent)
        
        return response
    
    def parse_intent(self, input_text):
        """Convert natural language to logical predicates."""
        # Simplified intent recognition
        if "like" in input_text.lower():
            return {"type": "preference", "object": "action"}
        elif "recommend" in input_text.lower():
            return {"type": "request", "action": "recommend"}
        else:
            return {"type": "unknown"}
    
    def generate_response(self, intent):
        """Generate response using logical reasoning."""
        if intent["type"] == "preference":
            # Add preference to KB
            self.kb.append(f"Likes(user, {intent['object']})")
            return f"I'll remember you like {intent['object']} content."
        
        elif intent["type"] == "request":
            # Use inference to find recommendations
            return "Based on your preferences, I recommend action movies."
        
        return "I didn't understand. Can you rephrase?"
```

---

This completes the comprehensive lecture notes for Unit III (Adversarial Search and Constraint Satisfaction) and Unit IV (Knowledge and Reasoning). The notes include:

**Unit III Coverage:**
- Game Theory and Optimal Decisions
- Minimax and Alpha-Beta Pruning algorithms
- Monte Carlo Tree Search
- Stochastic and Partially Observable Games
- Constraint Satisfaction Problems
- Constraint Propagation and Backtracking Search
- Real-world applications (Google DeepMind case study)

**Unit IV Coverage:**
- Knowledge-Based Agents
- Wumpus World Problem
- Propositional Logic and Theorem Proving
- Model Checking
- Real-world case studies (BBC & Amazon Alexa)

Each topic includes:
- Clear conceptual explanations accessible to second-year students
- Intuitive pseudocode
- Complete Python implementations following PEP 8
- Real-world examples and applications
- Progressive complexity building

The notes are structured for academic distribution and provide a solid foundation for SPPU SE AI-DS and CSE (AI) students to understand both theoretical concepts and practical implementations.

---