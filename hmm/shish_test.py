from hmm.hmm import normalize_rows, normalize_row
import numpy
import math 
from drs import drs


# 3.31
def test_normalize_rows_1D():
    
    # random_1D_array = drs(6, 100, [100]*6, [0]*6)
    random_1D_array = numpy.random.rand(6)
    print(random_1D_array)
    normalized_1D_array = normalize_rows(random_1D_array)
    assert sum(normalized_1D_array) == 1