import matplotlib.pyplot as plt
import numpy as np
from solver_time_tracker import solver_time_tracker

def plot_q_training_results(outcomes):
    x = np.arange(len(outcomes))
    labels = {1: 'win', 0.5: 'draw', -1: 'loss'}
    y = [labels[outcome] for outcome in outcomes]

    plt.scatter(x, y)
    plt.xlabel('Iteration')
    plt.ylabel('Outcome')
    plt.title('Q Training Results')
    plt.show()

def plot_benchmark_results(solvers, wins1, wins2, draws, prefix = ''):
    total_wins1 = sum(wins1)
    total_wins2 = sum(wins2)
    total_draws = sum(draws)
    
    if(solvers[0] == solvers[1]):
        solvers[0] += ' 1'
        solvers[1] += ' 2'

    categories = [f'{solvers[0]} Wins', f'{solvers[1]} Wins', 'Draws']
    values = [total_wins1, total_wins2, total_draws]
    
    plt.figure(figsize=(8, 6))
    
    bars = plt.bar(categories, values, color=['blue', 'green', 'gray'])
    
    plt.xlabel('Categories')
    plt.ylabel('Number of Outcomes')
    plt.title(f'Benchmark bar plot for {solvers[0]} vs {solvers[1]} in {prefix} game')
    plt.xticks(np.arange(len(categories)), categories)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom')
    
    filename = './Plots/benchmark_results_' + prefix + '_'.join(solvers) + '.png'
    plt.savefig(filename)


def plot_avg_times(solver_time_tracker, prefix = ''):
    avg_times = solver_time_tracker.get_solvers_times_avg()
    solvers = list(avg_times.keys())
    times = list(avg_times.values())

    plt.figure(figsize=(8, 6))
    
    bars = plt.bar(solvers, times, color='blue')
    
    plt.xlabel('Solvers')
    plt.ylabel('Average Time (s)')
    plt.title('Average time taken by each solver')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 2), va='bottom')
    
    filename = './Plots/solving_time_results_' + prefix + '_'.join(solvers) + '.png'
    plt.savefig(filename)

def print_benchmark_status(solvers, test, test_count):
    print(f"\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n[{test}/{test_count}] {solvers[0].__name__} vs {solvers[1].__name__}\n")

def print_training_status(episodes, episode, epsilon):
    fraction = episode / episodes
    fraction *= 10
    string = '=' * int(fraction)
    string += '>'
    string += ' ' * (10 - int(fraction))
    print(f"\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n[{string}] {episode}/{episodes} - Epsilon: {epsilon}\n")