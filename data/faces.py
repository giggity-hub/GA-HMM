import os
from natsort import natsorted
import json
from hmm.params import multiple_observation_sequences_from_ndarray_list


class FacesDataset:
    def __init__(self, folder='16bit') -> None:
        dir = os.path.join('./data/orl/', folder)
        self.observations = self.load_json_files_as_list(dir)
        

    def load_json_files_as_list(self, folder_dir):
        filenames = natsorted(os.listdir(folder_dir))
        file_paths = [os.path.join(folder_dir, filename) for filename in filenames]

        res = []
        for filepath in file_paths:
            with open(filepath, 'r') as f:
                data = json.load(f)
                res.append(data)
        
        return res


    def get_first_n_observations(self, category, n_observations):
        samples = self.observations[category][:n_observations]
        return multiple_observation_sequences_from_ndarray_list(samples)