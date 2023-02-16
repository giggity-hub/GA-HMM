# Hier in dieser File soll auch die Multiple Observations entstehen
import os
from natsort import natsorted
import json
import numpy
from typing import List, NamedTuple, Literal

OBSERVATIONS_DIR = 'data/observations/'

class MultipleObservationSequences(NamedTuple):
    """_summary_
    """
    slices: numpy.ndarray
    arrays: numpy.ndarray
    length: int


def observations(ndarray_list: List[numpy.ndarray]) -> MultipleObservationSequences:
    indices = numpy.zeros((len(ndarray_list) + 1), dtype=int)

    for i in range(len(ndarray_list)):
        indices[i + 1] = indices[i] + len(ndarray_list[i])
    
    unified_array = numpy.concatenate(ndarray_list, dtype=int)
    
    return MultipleObservationSequences(slices=indices, arrays=unified_array, length=len(ndarray_list))


class Observations:
    def __init__(self, dataset: Literal['fsdd', 'orl'], n_symbols: int) -> None:
        self.dataset = dataset
        self.n_symbols = n_symbols

        self._init_observation_paths()
        self._init_observations_for_category()
        

    def _init_observation_paths(self):
        observations_dir = os.path.join(OBSERVATIONS_DIR, self.dataset, str(self.n_symbols))
        observation_filenames = natsorted(os.listdir(observations_dir))
        observation_paths = [os.path.join(observations_dir, filename) for filename in observation_filenames]

        self.n_categories = len(observation_paths)
        self.observation_paths = observation_paths


    def _init_observations_for_category(self):
        observations_for_category = []
        for path in self.observation_paths:
            with open(path, 'r') as f:
                observations = json.load(f)
                observations_for_category.append(observations)

        self.observations_for_category = observations_for_category

    def get_first_n_observations_of_category(self, category_index: int, n: int, offset: int = 0):
        observations_list = self.observations_for_category[category_index][offset:(offset + n)]
        observations_tuple = observations(observations_list)
        return observations_tuple

    def get_first_n_observations_except_category(self, category_index:int, n:int, offset: int=0):
        observations_list = []

        for i in range(self.n_categories):
            if i != category_index:
                observations_list += self.observations_for_category[i][offset: (offset + n)]

        observations_tuple = observations(observations_list)
        return observations_tuple


