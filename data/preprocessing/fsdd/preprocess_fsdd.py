from natsort import natsorted
import os
from typing import List, Callable
from quantize import create_mfcc_codebook_from_audio_filepaths, quantize_audio_filepaths
from scipy.cluster.vq import vq
import json
from numpy import ndarray


def get_digit_from_filepath(filepath: str) -> int:
    filename = os.path.basename(filepath)
    filename_without_extension, _ = os.path.splitext(filename)
    digit, _, _ = filename_without_extension.split('_')
    return int(digit)


def get_all_filepaths_from_dir(dir):
    filenames = natsorted(os.listdir(dir))
    filepaths = [os.path.join(dir, filename) for filename in filenames]
    return filepaths


def sort_filepaths_into_categories(
    filepaths: List[str], 
    n_categories: int, 
    get_category_from_filepath: Callable[[str], int]
    ) -> List[List[str]]:

    categories = [[] for i in range(n_categories)]

    for filepath in filepaths:
        category = get_category_from_filepath(filepath)
        categories[category].append(filepath)

    return categories

OBSERVATIONS_DIR = './data/observations/'

def save_observations_as_json(observations_arr: List[ndarray], dataset, n_symbols, category):
    path = os.path.join(OBSERVATIONS_DIR, dataset, n_symbols)
    if not os.path.exists(path):
        os.makedirs(path)

    filename = f'{category}.json'
    filepath = os.path.join(path, filename)

    with open(filepath, 'w') as f:
        observations_list = [arr.tolist() for arr in observations_arr]
        json_data = json.dumps(observations_list)
        f.write(json_data)
    

FSDD_DIR = './data/raw/fsdd'
N_FSDD_CATEGORIES = 10

N_SYMBOLS = 256

fsdd_file_paths = get_all_filepaths_from_dir(FSDD_DIR)
filepaths_for_digit = sort_filepaths_into_categories(fsdd_file_paths, N_FSDD_CATEGORIES, get_digit_from_filepath)

for digit in range(N_FSDD_CATEGORIES):
    filepaths = filepaths_for_digit[digit]
    codebook = create_mfcc_codebook_from_audio_filepaths(filepaths, codebook_size=N_SYMBOLS)
    observations = quantize_audio_filepaths(filepaths, codebook)

    save_observations_as_json(observations, dataset='fsdd', n_symbols=str(N_SYMBOLS), category=str(digit))

    