from data.faces import FacesDataset
import pytest

face_numbers = [i for i in range(40)]
@pytest.mark.parametrize('face_number', face_numbers)
def test_faces_dataset(face_number):
    data = FacesDataset()
    observations = data.get_first_n_observations('8bit', face_number)
    assert observations.arrays.min() >= 0
    assert observations.arrays.max() <= 7

