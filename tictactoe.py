import QLearning
from copy import deepcopy
import plotting

###################################
players = ['X', 'O']

def get_available_moves(board):
    available_moves = []
    for row in range(3):
        for col in range(3):
            if board[row][col] == ' ':
                available_moves.append((row, col))
    return available_moves

def switch_players_on_board(board):
    for row in range(3):
        for col in range(3):
            if board[row][col] == 'X':
                board[row][col] = 'O'
            elif board[row][col] == 'O':
                board[row][col] = 'X'

def get_switched_board(board):
    new_board = deepcopy(board)
    switch_players_on_board(new_board)
    return new_board

def create_board():
    return [[' ' for _ in range(3)] for _ in range(3)]

def drop_piece(board, row, col, player):
    if row < 0 or row > 2 or col < 0 or col > 2:
        return False
    board[row][col] = player

def is_valid_location(board, row, col):
    if row < 0 or row > 2 or col < 0 or col > 2:
        return False
    return board[row][col] == ' '

def check_winner(board, player):
    for row in board:
        if all(cell == player for cell in row):
            return True
    for col in range(3):
        if all(board[row][col] == player for row in range(3)):
            return True
    if all(board[i][i] == player for i in range(3)) or all(board[i][2-i] == player for i in range(3)):
        return True
    return False

def get_winner(board):
    for player in players:
        if check_winner(board, player):
            return player
    return None

def check_draw(board):
    for row in board:
        if " " in row:
            return False
    return True

def print_board(board):
    for row in board:
        print("|".join(row))
        print("-----")

def solve_input(player, board):
    print(f"Player {player}'s turn")
    row, col = map(int, input("Enter row and column (0, 1, 2): ").split())
    while not is_valid_location(board, row, col):
        row, col = map(int, input("Enter row and column (0, 1, 2): ").split())
    return row, col

def next_move(solving_function, player, board):
    row, col = solving_function(player, board)
    drop_piece(board, row, col, player)

def get_oponent(player):
    if(player == 'X'):
        return 'O'
    return 'X'

#############################################

import random

def find_winning_move(board, player):
    for row in range(3):
        for col in range(3):
            if board[row][col] == ' ':
                board[row][col] = player
                if check_winner(board, player):
                    board[row][col] = ' '  # Undo the move
                    return (row, col)
                board[row][col] = ' '  # Undo the move
    return None

def find_blocking_move(board, player):
    opponent = 'O' if player == 'X' else 'X'
    return find_winning_move(board, opponent)

def find_random_move(board):
    valid_moves = get_available_moves(board)
    if valid_moves:
        return random.choice(valid_moves)
    return None

def solve_smart_random(player, board):
    move = find_winning_move(board, player)
    if move:
        return move

    move = find_blocking_move(board, player)
    if move:
        return move

    move = find_random_move(board)
    if move:
        return move

    return None


#############################################
 
def minimax(board, depth, is_maximizing, player):
    opponent = get_oponent(player)
    winner = get_winner(board)
    if winner == player:
        return 10 - depth
    elif winner == opponent:
        return depth - 10
    elif check_draw(board):
        return 0

    if is_maximizing:
        best_score = -float('inf')
        for row in range(3):
            for col in range(3):
                if is_valid_location(board, row, col):
                    board[row][col] = player
                    score = minimax(board, depth + 1, False, player)
                    board[row][col] = ' '
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for row in range(3):
            for col in range(3):
                if is_valid_location(board, row, col):
                    board[row][col] = opponent
                    score = minimax(board, depth + 1, True, player)
                    board[row][col] = ' '
                    best_score = min(score, best_score)
        return best_score

def solve_minmax(player, board):
    best_score = -float('inf')
    best_move = None
    for row in range(3):
        for col in range(3):
            if is_valid_location(board, row, col):
                board[row][col] = player
                score = minimax(board, 0, False, player)
                board[row][col] = ' '
                if score > best_score:
                    best_score = score
                    best_move = (row, col)
    return best_move

#############################################

def minimax_with_alpha_beta(board, depth, alpha, beta, is_maximizing, player):
    opponent = get_oponent(player)
    winner = get_winner(board)
    if winner == player:
        return 10 - depth
    elif winner == opponent:
        return depth - 10
    elif check_draw(board):
        return 0

    if is_maximizing:
        max_eval = -float('inf')
        for row in range(3):
            for col in range(3):
                if is_valid_location(board, row, col):
                    board[row][col] = player
                    eval = minimax_with_alpha_beta(board, depth + 1, alpha, beta, False, player)
                    board[row][col] = ' '
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break  # Beta cut-off
        return max_eval
    else:
        min_eval = float('inf')
        for row in range(3):
            for col in range(3):
                if is_valid_location(board, row, col):
                    board[row][col] = opponent
                    eval = minimax_with_alpha_beta(board, depth + 1, alpha, beta, True, player)
                    board[row][col] = ' '
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break  # Alpha cut-off
        return min_eval

def solve_minmax_alpha_beta(player, board):
    best_score = -float('inf')
    best_move = None
    alpha = -float('inf')
    beta = float('inf')
    for row in range(3):
        for col in range(3):
            if is_valid_location(board, row, col):
                board[row][col] = player
                score = minimax_with_alpha_beta(board, 0, alpha, beta, False, player)
                board[row][col] = ' '
                if score > best_score:
                    best_score = score
                    best_move = (row, col)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
    return best_move

#############################################

def print_training_status(episodes, episode, epsilon):
    fraction = episode / episodes
    fraction *= 10
    string = '=' * int(fraction)
    string += '>'
    string += ' ' * (10 - int(fraction))
    print(f"\n\n\n\n\n\n\n\n[{string}] {episode}/{episodes} - Epsilon: {epsilon}\n")

def train_q_learning(episodes=10):
    players = ['X', 'O']  # Assuming 'X' is the agent
    agent = QLearning.QLearning()
    agent.setMode('tictactoe')
    outcomes = []
    for _ in range(episodes):
        print_training_status(episodes, _, agent.epsilon)
        turn = 0
        board = create_board()
        game_over = False
        state = str(board)
        while not game_over:
            if(players[turn] == 'O'):
                row, col = solve_smart_random(players[turn], board)
                drop_piece(board, row, col, players[turn])
            else:
                state = str(board)
                possible_actions = get_available_moves(board)
                action = agent.solve_q_learning(players[turn], board)
                row, col = action
                drop_piece(board, row, col, players[turn])
                next_state = str(board)

            if check_winner(board, players[turn]):
                outcome = 1 if players[turn] == 'X' else -1
                outcomes.append(outcome)
                game_over = True
            elif check_draw(board):
                outcome = 0.5
                outcomes.append(0.5)
                game_over = True
            else:
                outcome = 0

            # If 'X' just played and won/lost/drew, update Q-table
            if game_over:
                agent.update_q_table(state, action, outcome, next_state, possible_actions)

            turn = (turn + 1) % 2
            if game_over:
                agent.update_epsilon()
    plotting.plot_q_training_results(outcomes)
    agent.save_model(f"./QOutput/q_learning_model_tictactoe.pkl")

#############################################

def play_game(solvers, debug=False):
    board = create_board()
    turn = 0

    while True:
        if debug:
            print_board(board)
        next_move(solvers[turn], players[turn], board)
        if check_winner(board, players[turn]):
            if(debug):
                print_board(board)
                print(f"Player {players[turn]} wins!")
            return players[turn]
        elif check_draw(board):
            if(debug):
                print_board(board)
                print("It's a draw!")
            return None

        turn = (turn + 1) % 2

def benchmarkAll(solvers, test_count=10):
    for i in range(len(solvers)):
        benchmarkSolvers(solvers[i], solvers[i], test_count)
        for j in range(i+1, len(solvers)):
            benchmarkSolvers(solvers[i], solvers[j], test_count)
    

def benchmarkSolvers(solver1, solver2, test_count=10):
    solvers = [solver1, solver2]
    wins = [0, 0]
    draws = 0
    for _ in range(test_count):
        winner = play_game(solvers)
        if winner == 'X':
            wins[0] += 1
        elif winner == 'O':
            wins[1] += 1
        else:
            draws += 1

    plotting.plot_benchmark_results([solver1.__name__, solver2.__name__], [wins[0]], [wins[1]], [draws], 'tictactoe')

isTraining = False
QTRAINING_ITERATIONS = 100000
BENCHMARK_TEST_COUNT = 100

def main():

    if isTraining:
        train_q_learning(QTRAINING_ITERATIONS)
    else:
        agent = QLearning.QLearning()
        agent.load_model("./QOutput/q_learning_model_tictactoe.pkl")
        solvers = [solve_smart_random, solve_minmax, solve_minmax_alpha_beta, agent.solve_q_learning]
        winner = play_game(solvers)
        benchmarkAll(solvers, BENCHMARK_TEST_COUNT)

if __name__ == "__main__":
    main()
