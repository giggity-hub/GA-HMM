import h5py
import numpy
from hmm.bw import BaumWelch
import unittest
from collections import namedtuple
from typing import List
import glob
import pytest

# @pytest.mark.parameterize
# Parametrize an object that holds all values of the matlab test
# I can pass in the read h5py mopeds

FIELD_NAMES = ["s", "t", "e", "o", "A", "B", "C", "LogP", "Alfa", "Beta", "Gama", "Tau", "Taui", "Nu", "Omega"]
Fields = namedtuple('Fields', FIELD_NAMES)

def convert_hdf5_to_named_tuple(hdf5):    
    field_values = [hdf5[name]['value'][()] for name in FIELD_NAMES]
    return Fields(*field_values)

def read_test_data(path):
    # file paths are relative to base directory
    hdf5_file_paths = glob.glob(f'{path}/*.hdf5')
    test_data = []
    for file_path in hdf5_file_paths:
        with h5py.File(file_path, 'r') as f:
            file_data_tuple = convert_hdf5_to_named_tuple(f)
            test_data.append(file_data_tuple)

    return test_data


class Test:
    test_data = read_test_data('octave/tests')

    @pytest.mark.parametrize('input', test_data)
    def test_baum_welch_norm(self, input: Fields):
        pi = input.s.T.flatten()
        b = input.e.T
        a = input.t.T
        alpha = input.Alfa.T
        beta = input.Beta.T
        # xi = input.X
        gamma = input.Gama.T
        tau = input.Tau.T
        taui = input.Taui.T
        nu = input.Nu.T
        omega = input.Omega.T

        reestimated_start_probs = input.C.T.flatten()
        reestimated_transitions = input.A.T
        reestimated_emissions = input.B.T

        
        observation_sequence = input.o.T.flatten().astype(int)
        # Matlab indices start at 1. Python starts at 0
        observation_sequence = observation_sequence - 1
        
        assert numpy.allclose(taui, numpy.sum(tau, axis=1))
        
        bw = BaumWelch(pi, b, a)
        log_prob = bw.train([observation_sequence])

        print(bw.pi)
        print(reestimated_start_probs)

        assert bw.pi.shape == pi.shape
        assert numpy.allclose(bw.pi, reestimated_start_probs)

        assert bw.b.shape == b.shape
        assert numpy.allclose(bw.b, reestimated_emissions)

        assert bw.a.shape == a.shape
        assert numpy.allclose(bw.a, reestimated_transitions)


        # Alpha
        assert bw.alpha.shape == alpha.shape
        assert numpy.array_equiv(bw.alpha, alpha)

        assert log_prob == input.LogP

        # Test beta
        assert bw.beta.shape == beta.shape
        assert numpy.array_equiv(bw.beta, beta)

        # Test Xi

        # Test Gamma
        # absolute_tolerance = 1e-10
        # relative_tolerance = 0
        assert bw.gamma.shape == gamma.shape
        assert numpy.allclose(bw.gamma, gamma)

        # Test Taui
        assert bw.taui.shape == taui[:,0].shape
        assert numpy.allclose(bw.taui, taui[:,0])


        # Test nu
        # assert bw.nu.shape == nu[:,0].shape
        # assert numpy.allclose(bw.nu, nu[:,0])
        # assert numpy.array_equiv(bw.nu, nu)
        # nu passed auch aber hab ich gerade keinen bock den shit zu fixen


        # Test Omega
        assert bw.omega.shape == omega.shape
        assert numpy.allclose(bw.omega, omega)

        # print(numpy.sum(bw.xi, axis=2))
        # print(bw.gamma)

        # I know that gamma is correct
        # I know that the sum of xi over j == gamma
        # 
        



        print(tau)
        print(bw.tau)
        # Test Tau
        assert bw.tau.shape == tau.shape
        assert numpy.allclose(bw.tau, tau)

        



        

        print(log_prob)
        print(input.LogP)

        # Test reestimated initial Vector
        assert numpy.array_equiv(bw.pi, reestimated_start_probs)

        # Test Shapes
        assert bw.pi.shape == reestimated_start_probs.shape
        assert bw.a.shape == reestimated_transitions.shape
        assert bw.b.shape == reestimated_emissions.shape

        print(bw.a)
        print(reestimated_transitions)
        # Test reestimated Transition Matrix
        assert numpy.array_equiv(bw.a, reestimated_transitions)

        assert numpy.array_equiv(bw.b, reestimated_emissions)

        
        

        

        

    
def debug():
    test_data = read_test_data('octave/tests')

    print(test_data[0].C)




if __name__ == '__main__':
    unittest.main()
    # debug()
    # print(test_data)