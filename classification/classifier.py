from ga.ga import GeneticAlgorithm
from typing import List, Union

class Classifier:
    def __init__(self, 
        ga_instance: GeneticAlgorithm,
        hmm_n_states: int,
        hmm_alphabet: List[int],
        n_classes: int, 
        class_names: List[str]=None ) -> None:

        self.n_classes = n_classes

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

    def train_bw(self):
        hmm_dict = dict.fromkeys(self.class_names, None)

        # RandomHmm(n,m)
        # Initialize the dict with random hmms
        # for cl_name in self.class_names:

            