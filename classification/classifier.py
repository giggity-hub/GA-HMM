from ga.ga import GeneticAlgorithm
from typing import List, Union, Callable, Tuple, Dict
import numpy
from hmm.hmm import hmm_from_params

class Classifier:
    def __init__(self, 
        ga_instance: GeneticAlgorithm,
        hmm_n_states: int,
        hmm_alphabet: List[int],
        n_classes: int, 
        hmm_param_generator: Callable[[], Tuple[numpy.array, numpy.ndarray, numpy.ndarray]],
        class_names: List[str]=None ) -> None:

        self.n_classes = n_classes
        self.hmm_param_generator = hmm_param_generator

        if not class_names:
            class_names = [str(i) for i in range(n_classes)]
        self.class_names = class_names
        


    def train_ga(self):
        # benötigt
        # initiale population von genes
        # Bei der Initialen Population muss ich dann darauf achten wie ich die lr matrix richtig instantiate
        # => obserste prio: wie random lr hmm und lr hmm genetic representation erzeugen?
        # ich kann einfach nur 
        pass 

    def train_bw(self, sequences: Dict[str, List[List[Union[str, int]]]]):
        """_summary_

        Args:
            sequences (Dict[str, List[List[Union[str, int]]]]): 

        Returns:
            _type_: _description_
        """
        
        hmm_dict = dict.fromkeys(self.class_names, None)
        
        for cl_name in self.class_names:
            hmm_params = self.hmm_param_generator()
            model = hmm_from_params(*hmm_params)
            model.fit(sequences[cl_name])
            hmm_dict[cl_name] = model
            

        
        return hmm_dict
        

        
            

        # RandomHmm(n,m)
        # Initialize the dict with random hmms
        # for cl_name in self.class_names:

            