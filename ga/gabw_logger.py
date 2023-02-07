import numpy
import seaborn as sns
import matplotlib.pyplot as plt
from collections import namedtuple

training_types_tuple = namedtuple('types', 'GA BW')

class GABWLogger:

    _nth_iteration_was_perfored_with: numpy.ndarray
    training_types = training_types_tuple(0, 1)

    def __init__(self, n_hmms, n_log_entries) -> None:
        
        self.current_log_index = 0
        self.logs = numpy.empty((n_hmms, n_log_entries))
        self._nth_iteration_was_perfored_with = numpy.empty(n_log_entries)
        
        
    def _insert(self, log_sequences, was_performed_with: int):
        n_logs = log_sequences.shape[1]

        start = self.current_log_index
        stop = self.current_log_index + n_logs

        self.logs[:, start: stop] = log_sequences
        self._nth_iteration_was_perfored_with[start:stop] = was_performed_with

        self.current_log_index += n_logs

    def insert_ga_iterations(self, log_sequencess):
        self._insert(log_sequencess, self.training_types.GA)

    def insert_bw_iterations(self, log_sequences):
        self._insert(log_sequences, self.training_types.BW)

    def plot(self):
        g = sns.scatterplot(data=self.logs.T, legend=False, markers='.')
        # label_names = self.training_types._fields
        # plt.legend(title='Smoker', loc='upper left', labels=label_names)
        plt.show(g)