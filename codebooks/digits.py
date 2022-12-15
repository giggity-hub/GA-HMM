import os
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from scipy.spatial import distance
# from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans2, vq, kmeans
from data.digits import get_digits
import librosa
import numpy as np

class DigitsCodebook:
    def __init__(self, file_paths, codebook_size=128) -> None:

        self.codebook_size=128
        self.mfcc_feature_dims = 13
        self.mfcc_features = [[] for i in range(10)]
        self.codebooks = [[] for i in range(10)]


        # read wav files
        for digit, digit_file_paths in enumerate(file_paths):
            # create Mfcc features
            for file_path in digit_file_paths:
                (rate,sig) = wav.read(file_path)
                mfcc_feat = mfcc(sig,rate)
                self.mfcc_features[digit].append(mfcc_feat)

                
        # Create list with codebooks
        for digit in range(10):
            digit_features = np.vstack(self.mfcc_features[digit])
            centroids, distortion = kmeans(digit_features, self.codebook_size)
            self.codebooks[digit] = centroids

        # for digit, digit_mfcc_matrices in enumerate(self.mfcc_features):
        #     digit_mfcc_features = [vec for matrix in digit_mfcc_matrices for vec in matrix]
        #     centroids = kmeans(digit_mfcc_features, self.codebook_size)
        #     self.codebooks[digit] = centroids



    def quantize_from_path(self, digit, wav_path):
        rate, sig = wav.read(wav_path)
        mfcc_features = mfcc(sig,rate)
        quantized, distortion = vq(mfcc_features,self.codebooks[digit])
        return quantized

    def quantize_from_path_list(self, digit, wav_path_list):
        return [self.quantize_from_path(digit, wp) for wp in wav_path_list]



if __name__ == "__main__":

    training_file_paths = get_digits(dataset="train", n_digits=10)
    print(training_file_paths)
    
    # codebook = DigitsCodebook(training_file_paths)

    

    


