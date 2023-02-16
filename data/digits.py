import os
import json
from hmm.params import multiple_observation_sequences_from_ndarray_list

# Returns 2D Array with Filepaths for digits
def get_digits(dataset="train", n_digits=float('inf'), digits=[0,1,2,3,4,5,6,7,8,9] ):
    data_dir = f"./data/fsdd/{dataset}/"
    training_file_paths = []

    for i in range(10):
        digit_folder = f"{data_dir}{i}/"
        digit_file_names = os.listdir(digit_folder)
        file_count = len(digit_file_names)
        return_file_count = int(min(n_digits, file_count))

        digit_file_paths = [digit_folder + file_name for file_name in digit_file_names][:return_file_count]
        training_file_paths.append(digit_file_paths)

    return training_file_paths


def load_dataset(dataset="train"):
    valid_datasets = ["train", "test", "validate"]

    if not dataset in valid_datasets: 
        raise Exception("specified dataset does not exist. Must be of Value: 'train', 'test' or 'validate'")

    data_dir = os.path.join("./data/quantized/", dataset)
    res = []

    for digit in range(10):
        filename = f'{digit}.json'
        json_path = os.path.join(data_dir, filename)
        with open(json_path, 'r') as f:
            digit_samples = json.load(f)
            res.append(digit_samples)

    return res

class DigitDataset:
    def __init__(self, dataset="train") -> None:
        valid_datasets = ["train", "test", "validate"]

        if not dataset in valid_datasets: 
            raise Exception("specified dataset does not exist. Must be of Value: 'train', 'test' or 'validate'")

        self.data_dir = os.path.join("./data/quantized/", dataset)
        self.observations = self._load()
        
    def _load(self):
        res = []
        for digit in range(10):
            filename = f'{digit}.json'
            json_path = os.path.join(self.data_dir, filename)
            with open(json_path, 'r') as f:
                digit_samples = json.load(f)
                res.append(digit_samples)

        return res
    
    def get_first_n_observations(self, category, n_observations):
        samples = self.observations[category][:n_observations]
        return multiple_observation_sequences_from_ndarray_list(samples)



# class Data:
#     def __init__(self) -> None:

#         pass

#     def training_samples(self, n_items):


#     def test_samples(digit: int):
