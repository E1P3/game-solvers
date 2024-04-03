import matplotlib.pyplot as plt
import numpy as np

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
    # Assuming wins1, wins2, and draws are already aggregated totals
    # If they're not, you can sum them up here. For example:
    # total_wins1 = sum(wins1)
    # total_wins2 = sum(wins2)
    # total_draws = sum(draws)

    total_wins1 = sum(wins1)
    total_wins2 = sum(wins2)
    total_draws = sum(draws)
    
    if(solvers[0] == solvers[1]):
        solvers[0] += ' 1'
        solvers[1] += ' 2'

    categories = [f'{solvers[0]} Wins', f'{solvers[1]} Wins', 'Draws']
    values = [total_wins1, total_wins2, total_draws]
    
    plt.figure(figsize=(8, 6))
    
    # Plotting
    bars = plt.bar(categories, values, color=['blue', 'green', 'gray'])
    
    # Adding some aesthetics
    plt.xlabel('Categories')
    plt.ylabel('Number of Outcomes')
    plt.title(f'Benchmark bar plot for {solvers[0]} vs {solvers[1]} in {prefix} game')
    plt.xticks(np.arange(len(categories)), categories)
    
    # Adding value labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom')  # va: vertical alignment
    
    filename = './Plots/benchmark_results_' + prefix + '_'.join(solvers) + '.png'
    plt.savefig(filename)
