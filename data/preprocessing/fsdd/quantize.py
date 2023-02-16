from python_speech_features import mfcc
import scipy.io.wavfile as wav
from scipy.cluster.vq import vq, kmeans
import numpy as np
from typing import List


def create_mfcc_codebook_from_audio_filepaths( filepaths: List[str], codebook_size: int) -> np.ndarray:
    
    all_mfcc_features_list = []

    for filepath in filepaths:
        (rate, sig) = wav.read(filepath)
        mfcc_features_of_file = mfcc(sig,rate)
        all_mfcc_features_list.append(mfcc_features_of_file)

    all_mfcc_features_array = np.vstack(all_mfcc_features_list)
    centroids, _ = kmeans(all_mfcc_features_array, codebook_size)

    return centroids

def quantize_audio_filepaths(filepaths: List[str], codebook):
    quantized_mfcc_features = []

    for filepath in filepaths:
        (rate, sig) = wav.read(filepath)
        mfcc_features_of_file = mfcc(sig,rate)
        quantized, _ = vq(mfcc_features_of_file, codebook)
        quantized_mfcc_features.append(quantized)

    return quantized_mfcc_features