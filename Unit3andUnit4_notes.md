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