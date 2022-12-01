from hmm.hmm import normalize_rows
import numpy

ERROR_TOLERANCE = 5e-16

def within_tolerance(a,b):
    return abs(a - b) <= ERROR_TOLERANCE


def test_normalize_rows_1D():
    random_1D_array = numpy.random.rand(6)
    normalized_1D_array = normalize_rows(random_1D_array)
    assert within_tolerance(sum(normalized_1D_array), 1)


def test_normalize_rows_2D():
    random_2D_array = numpy.random.rand(6,8)
    normalized_2D_array = normalize_rows(random_2D_array)
    row_sums = normalized_2D_array.sum(axis=1)
    for r_sum in row_sums:
        assert within_tolerance(r_sum, 1)