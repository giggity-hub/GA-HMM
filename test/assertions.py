import math
import numpy

def assert_is_log_prob(log_prob):
    assert type(log_prob) == float
    assert not math.isnan(log_prob)
    assert log_prob < 0

def assert_no_nan_values(ndarr):
    assert not numpy.any(numpy.isnan(ndarr))

def assert_all_values_leq_one(ndarr):
    assert numpy.all(numpy.less_equal(ndarr, 1))

def assert_all_values_geq_zero(ndarr):
    assert numpy.all(numpy.greater_equal(ndarr, 0))


def assert_all_values_are_probabilities(ndarr):
    assert_no_nan_values(ndarr)
    assert_all_values_geq_zero(ndarr)
    assert_all_values_leq_one(ndarr)

def assert_all_values_lt_zero(ndarr):
    assert numpy.all(numpy.less(ndarr, 0))

def assert_all_values_are_log_probabilities(ndarr):
    assert_no_nan_values(ndarr)
    assert_all_values_lt_zero(ndarr)