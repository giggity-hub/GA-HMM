import pytest



data = [0, 1, 2, 3]
@pytest.fixture(params=data)
def arg1(request):
    return request.param

@pytest.fixture
def arg2(arg1):
    return arg1 * 10

def test_args(arg1, arg2):
    print(arg1)
    print(arg2)
    assert False