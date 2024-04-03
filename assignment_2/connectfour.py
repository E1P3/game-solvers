import random
import math
import copy

def create_board():
    return [[' ' for _ in range(7)] for _ in range(6)]

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    return board[5][col] == ' '

def get_available_moves(board):
    available_moves = []
    for col in range(7):
        if is_valid_location(board, col):
            available_moves.append(col)
    return available_moves

def get_next_open_row(board, col):
    for r in range(6):
        if board[r][col] == ' ':
            return r

def print_board(board):
    for row in reversed(board):
        print("|".join(row))
        print("-" * 13)


def winning_move(board, piece):
    for r in range(6):
        for c in range(4):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

    for r in range(3):
        for c in range(7):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

    for r in range(3):
        for c in range(4):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True

    for r in range(3, 6):
        for c in range(4):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True

def next_move(solving_function, player, board):
    col = solving_function(player, board)
    row = get_next_open_row(board, col)
    drop_piece(board, row, col, player)

def get_opponent(player):
    if(player == 'X'):
        return 'O'
    return 'X'

def check_draw(board):
    return all(board[5][col] != ' ' for col in range(7))
#############################################

def solve_input(player, board):
    print(f"Player {player}'s turn")
    col = int(input("Enter column (0-6): "))
    while not is_valid_location(board, col):
        col = int(input("Enter column (0-6): "))
    return col

#############################################

def solve_smart_random(player, board):
    columns = list(range(7))
    random.shuffle(columns) 

    for col in columns:
        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, player)
            if winning_move(board, player):
                board[row][col] = ' ' 
                return col
            board[row][col] = ' ' 

    for col in columns:
        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, get_opponent(player))
            if winning_move(board, get_opponent(player)):
                board[row][col] = ' ' 
                return col
            board[row][col] = ' ' 

    valid_columns = [col for col in range(7) if is_valid_location(board, col)]
    return random.choice(valid_columns) if valid_columns else None

#############################################

def evaluate_window(window, piece):

    opponent_piece = get_opponent(piece)

    score = 0
    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(' ') == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(' ') == 2:
        score += 2
    if window.count(opponent_piece) == 3 and window.count(' ') == 1:
        score -= 4

    return score

def score_position(board, piece):
    score = 0

    center_array = [row[3] for row in board]
    center_count = center_array.count(piece)
    score += center_count * 3

    for r in range(6):
        row_array = board[r]
        for c in range(4):
            window = row_array[c:c+4]
            score += evaluate_window(window, piece)

    for c in range(7):
        col_array = [board[r][c] for r in range(6)]
        for r in range(3):
            window = col_array[r:r+4]
            score += evaluate_window(window, piece)

    for r in range(3):
        for c in range(4):
            window = [board[r+i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)

    for r in range(3, 6):
        for c in range(4):
            window = [board[r-i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)

    return score


def is_terminal_node(board):
    return winning_move(board, 'X') or winning_move(board, 'O') or len(get_valid_locations(board)) == 0

def minimax_alpa_beta(board, depth, alpha, beta, maximizingPlayer, player):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, player):
                return (None, 100000000000000)
            elif winning_move(board, get_opponent(player)):
                return (None, -10000000000000)
            else: 
                return (None, 0)
        else: 
            return (None, score_position(board, player))
    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = copy.deepcopy(board)
            drop_piece(b_copy, row, col, player)
            new_score = minimax_alpa_beta(b_copy, depth-1, alpha, beta, True, player)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value
    else: 
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = copy.deepcopy(board)
            drop_piece(b_copy, row, col, get_opponent(player))
            new_score = minimax_alpa_beta(b_copy, depth-1, alpha, beta, False, player)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value

def minimax(board, depth, maximizingPlayer, player):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, player):
                return (None, float('inf'))
            elif winning_move(board, get_opponent(player)):
                return (None, float('-inf'))
            else:
                return (None, 0)
        else:
            return (None, score_position(board, player))
    
    if maximizingPlayer:
        value = float('-inf')
        column = random.choice(valid_locations) if valid_locations else None
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = copy.deepcopy(board)
            drop_piece(b_copy, row, col, player)
            new_score = minimax(b_copy, depth-1, False, player)[1]
            if new_score > value:
                value = new_score
                column = col
        return column, value
    else:
        value = float('inf')
        column = random.choice(valid_locations) if valid_locations else None
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = copy.deepcopy(board)
            drop_piece(b_copy, row, col, player)
            new_score = minimax(b_copy, depth-1, True, player)[1]
            if new_score < value:
                value = new_score
                column = col
        return column, value

def get_valid_locations(board):
    valid_locations = []
    for col in range(7):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations

def pick_best_move(board, piece):
    valid_locations = get_valid_locations(board)
    best_score = -10000
    best_col = random.choice(valid_locations)
    for col in valid_locations:
        row = get_next_open_row(board, col)
        temp_board = [row[:] for row in board]
        drop_piece(temp_board, row, col, piece)
        score = score_position(temp_board, piece)
        if score > best_score:
            best_score = score
            best_col = col
    return best_col

def solve_minimax_alpa_beta(player, board, depth=5):
    is_maximizing = False
    column, minimax_score = minimax_alpa_beta(board, depth, -math.inf, math.inf, is_maximizing, player)
    if column is None:
        column = get_valid_locations(board)[0] if get_valid_locations(board) else None
    return column

def solve_minimax(player, board, depth=4):
    is_maximizing = False
    column, minimax_score = minimax(board, depth, is_maximizing, player)
    if column is None:
        column = get_valid_locations(board)[0] if get_valid_locations(board) else None
    return column

#############################################

def play_game(solvers, debug=False):
    board = create_board()
    game_over = False
    players = ['X', 'O']
    turn = 0

    while not game_over:
        if debug:
            print_board(board)
        next_move(solvers[turn], players[turn], board)
        if winning_move(board, players[turn]):
            if debug:
                print_board(board)
                print(f"Player {players[turn]} wins!")
            game_over = True
        elif check_draw(board):
            if debug:
                print_board(board)
                print("It's a draw!")
            game_over = True
        turn += 1
        turn %= 2
    
    return players[turn]

def benchmarkAll(solvers):
    for i in range(len(solvers)):
        benchmarkSolvers(solvers[i], solvers[i])
        for j in range(i+1, len(solvers)):
            benchmarkSolvers(solvers[i], solvers[j])
    

def benchmarkSolvers(solver1, solver2, test_count=5):
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

    print("___________________________")
    print(f"{test_count} games played")
    print(f"{solver1} wins: {wins[0]}")
    print(f"{solver2} wins: {wins[1]}")
    print(f"Draws: {draws}")
    print("___________________________")


def main():
    solvers = [solve_smart_random, solve_minimax, solve_minimax_alpa_beta]
    #winner = play_game(solvers)
    benchmarkAll(solvers)

if __name__ == "__main__":
    main()
