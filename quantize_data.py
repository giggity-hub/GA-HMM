from data.digits import get_digits
from codebooks.digits import DigitsCodebook
import os
import json


# Step 1: Create Codebook from Training Data
training_file_paths = get_digits()
cb = DigitsCodebook(training_file_paths)

# Step 2: Quantize every wav file and store as json

SUBFOLDER_NAMES = ['test', 'train', 'validate']
QUANTIZED_DIR = 'data/quantized'
FSDD_DIR = 'data/fsdd'

os.makedirs(QUANTIZED_DIR, exist_ok=True)

for sf_name in SUBFOLDER_NAMES:
    for digit in range(10):
        in_dir = os.path.join(FSDD_DIR, sf_name, str(digit))
        in_file_names = os.listdir(in_dir)
        in_file_paths = map(lambda name: os.path.join(in_dir, name) , in_file_names)

        out_file_name = f'{digit}.json'
        out_dir = os.path.join(QUANTIZED_DIR, sf_name)
        out_file_path = os.path.join(out_dir, out_file_name)
        

        quantized_np_arrs = cb.quantize_from_path_list(digit, in_file_paths)
        
        quantized_lists = list(map(lambda x: x.tolist(), quantized_np_arrs))
        print(type(quantized_np_arrs[0]))

        os.makedirs(out_dir, exist_ok=True)
        with open(out_file_path, "w") as f:
            json.dump(quantized_lists, f)

