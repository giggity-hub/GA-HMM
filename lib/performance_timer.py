import timeit

class PerformanceTimer():
    def __init__(self) -> None:
        self.total_time = 0
        self.n_intervals = 0
        self._start_time = 0

    def start(self):
        self._start_time = timeit.default_timer()

    def stop(self):
        end_time = timeit.default_timer()
        self.total_time += end_time - self._start_time
        self.n_intervals += 1