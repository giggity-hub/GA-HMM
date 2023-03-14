import numpy
from data.data import observations


_OBSERVATION_SEQUENCES_LISTS = [
    [1,2,3,1,4,2,4,4],
    [1,2,2,1,1,1,2,2,2,1],
    [1,2,3,2,1,2,5,4,1,2,4],
    [1,1,1,2,2,1,2,3],
    [1,1,1,2,2,2,3,3,3],
    [1,2,3,1,2,3,1,2,3],
    [1,1,1,2,2,2,3,3,3,1,2,3],
    [1,1,2,2,3,3,4,4,1,2,3,4],
    [1,1,1,1,2,2,2,2],
    [1,2,3,4,5,6,6,5,4,3,2,1]
]

def load_observation_sequences():
    observation_sequences = []
    for i in range(len(_OBSERVATION_SEQUENCES_LISTS)):
        obs_list = _OBSERVATION_SEQUENCES_LISTS[i]
        obs_arr = numpy.array(obs_list)
        obs_arr_zero_based = obs_arr - 1
        obs_tuple = observations([obs_arr_zero_based])
        observation_sequences.append(obs_tuple)
    return observation_sequences 

OBSERVATION_SEQUENCES = load_observation_sequences()
N_SYMBOLS_FOR_OBSERVATION_SEQUENCE = [max(obs) for obs in _OBSERVATION_SEQUENCES_LISTS]