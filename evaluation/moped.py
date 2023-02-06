# %load_ext autoreload
# %autoreload 2

from hmm.hmm import random_left_right_hmm_params
import numpy as np
import hmm.bw as bw
from data.digits import DigitDataset
import matplotlib.pyplot as plt
import seaborn as sns


training_data = DigitDataset('train')
observations = training_data.get_first_n_observations(category=0, n_observations=13)

n_hmms = 50
n_bw_iterations = 100

n_states = 4
n_symbols = 128

all_hmm_params = [random_left_right_hmm_params(n_states, n_symbols) for i in range(n_hmms)]
log_probability_of_hmm_after_iteration = np.empty((n_hmms, n_bw_iterations))


for i in range(len(all_hmm_params)):
    hmm_params = all_hmm_params[i]
    reestimated_hmm_params, log_probaility_after_iteration = bw.train_multiple_observations(hmm_params, observations, n_bw_iterations)
    log_probability_of_hmm_after_iteration[i, :] = log_probaility_after_iteration