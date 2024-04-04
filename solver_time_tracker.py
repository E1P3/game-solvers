import time
import numpy as np

class solver_time_tracker:
    def __init__(self):
        self.solver_times = {} # Dictionary to store array of times taken by each solver for each run
        self.timer = None

    def start_timer(self):
        self.timer = time.time()

    def stop_timer(self, solver_name):
        if solver_name not in self.solver_times:
            self.solver_times[solver_name] = []
        self.solver_times[solver_name].append(time.time() - self.timer)

    def get_solvers_times_avg(self):
        return {solver: np.mean(times) for solver, times in self.solver_times.items()}