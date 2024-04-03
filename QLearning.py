import numpy as np
import pickle
import tictactoe
import connectfour

class QLearning:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01, q_table={}):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}  # Q-value table
        self.mode = None

    def update_q_table(self, state, action, reward, next_state, possible_actions):
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0
        
        next_max = max(self.q_table[next_state][a] if next_state in self.q_table and a in self.q_table[next_state] else 0 for a in possible_actions)
        self.q_table[state][action] += self.alpha * (reward + self.gamma * next_max - self.q_table[state][action])

    def choose_action(self, state, possible_actions, default_action=None):
        if np.random.uniform(0, 1) < self.epsilon or state not in self.q_table:
            if default_action is not None:
                return default_action
            # Explore: choose a random action by index
            action_index = np.random.randint(len(possible_actions))
            action = possible_actions[action_index]
        else:
            # Exploit: choose the action with the highest Q-value for the current state
            state_actions = self.q_table[state]
            # This assumes that `possible_actions` might not include all actions in `state_actions`.
            # Adjust accordingly if `possible_actions` always includes all actions.
            best_action = max(state_actions, key=lambda a: state_actions.get(a, 0) if a in possible_actions else float('-inf'))
            action = best_action if best_action in possible_actions else np.random.choice(possible_actions)
        return action

    def update_epsilon(self):
        """Apply epsilon decay, respecting the minimum value."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

    def get_q_value(self, state, action):
        return self.q_table.get(state, {}).get(action, 0)

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'mode': self.mode,
                'q_table': self.q_table
            }, f)
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.alpha = data['alpha']
            self.gamma = data['gamma']
            self.epsilon = data['epsilon']
            self.mode = data['mode']
            self.q_table = data['q_table']

    def setMode(self, mode):
        self.mode = mode

    def solve_q_learning(self, player, board):
        action = None

        if self.mode == 'tictactoe':
            if player == 'O':
                currentBoard = tictactoe.get_switched_board(board)
            else:
                currentBoard = board

            row, col = tictactoe.solve_smart_random(player, currentBoard)
            default_action = (row, col)
            state = str(currentBoard)
            possible_actions = tictactoe.get_available_moves(currentBoard)
            action = self.choose_action(state, possible_actions, default_action)

        if self.mode == 'connectfour':
            if player == 'O':
                currentBoard = connectfour.get_switched_board(board)
            else:
                currentBoard = board

            default_action = connectfour.solve_smart_random(player, currentBoard)
            state = str(currentBoard)
            possible_actions = connectfour.get_available_moves(currentBoard)
            action = self.choose_action(state, possible_actions, default_action)


        return action
    