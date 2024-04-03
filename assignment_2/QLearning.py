import numpy as np

class QLearning:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, q_table={}):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = q_table

    def get_q_value(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = np.zeros((3, 3, 9), dtype=float)
        return self.q_table[state][action]

    def update_q_value(self, state, action, new_q_value):
        if state not in self.q_table:
            self.q_table[state] = np.zeros((3, 3, 9), dtype=float)
        self.q_table[state][action] = new_q_value

    def choose_action(self, state, valid_moves):
        if np.random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random valid move
            return valid_moves[np.random.choice(len(valid_moves))]
        else:
            # Exploit: choose the move with the highest Q-value
            q_values = [self.get_q_value(state, move) for move in valid_moves]
            max_q_value = np.max(q_values)
            best_moves = [move for move, q_value in zip(valid_moves, q_values) if q_value == max_q_value]
            return best_moves[np.random.choice(len(best_moves))]

    def load_q_table(self, filename):
        self.q_table = np.load(filename, allow_pickle=True).item()

    def save_q_table(self, filename):
        np.save(filename, self.q_table)