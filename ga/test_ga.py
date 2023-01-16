import unittest
from ga.ga import GaHMM, Chromosome


class GaHmmTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.gabw = GaHMM(
            n_symbols=128,
            n_states=4,
            n_symbols=128,
            n_states=4,
            population_size=20,
            n_generations=200,
            fitness_func=fitness_func,
            parent_select_func=parent_select_func,
            mutation_func=mutation_func,
            crossover_func=crossover_func,
            keep_elitism=1
        )

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()