def hmm2genes(hmm: HiddenMarkovModel):

    # Step 1: Get that motherfucking transitions shit
    transitions = hmm.dense_transition_matrix()
    # delete silent states along both axis

    # remove the hidden states (start, end) from the emissions matrix
    transitions = numpy.delete(transitions, numpy.s_[hmm.silent_start:], axis=1)

    # The start probabilities are the outgoing probabilities from the hidden start state
    start_probs = transitions[hmm.start_index]

    transitions = numpy.delete(transitions, numpy.s_[hmm.silent_start:], axis=0)

    transitions = transitions.flatten()    

    # Step 2: Get the state emissions
    state_emissions = []
    for state in hmm.states[:hmm.silent_start]:
        # Wichtig! Funzt nur unter der Vorraussetzung, dass die Values in der Reihenfolge sind wie übergeben wurden
        state_emissions += state.distribution.values()
    
    
    genes = numpy.concatenate((
        start_probs,
        state_emissions,
        transitions
    ))

    return Individual(genes)

class Individual:
    n_states: int = 0
    n_symbols: int = 0
    n_genes: int = 0
    alphabet: List[int] = []
    param_generator_func: any = None
    ready: bool = False
    range_start_probs: tuple = (0,0)
    range_emission_probs: tuple = (0,0)
    range_transition_probs: tuple = (0,0)
    slice_start_probs: slice = slice(0,0)
    slice_emission_probs: slice = slice(0,0)
    slice_transition_probs: slice = slice(0,0)
    emission_probs_legal_cutpoints: List[int] = []
    transition_probs_legal_cutpoints: List[int] = []
    legal_cutpoints: List[int] = []
    
    def __init__(self, genes=[], normalize_genes=False):

        self.fitness: float = float('-inf')

        if len(genes) == 0:
            start_probs, transition_probs, emission_probs = self.param_generator_func(self.n_states, self.n_symbols)
            genes = numpy.concatenate((
                start_probs,
                transition_probs.flatten(),
                emission_probs.flatten()
            ))
        self.genes = genes

        if normalize_genes:
            self.normalize()

    @classmethod
    def initialize_class(cls, n_states: int, n_symbols, param_generator_func=random_left_right_hmm_params):
        cls.n_states = n_states
        cls.n_symbols = n_symbols
        cls.param_generator_func = staticmethod(param_generator_func)
        cls.alphabet = [i for i in range(n_symbols)]

        cls.n_genes = n_states + n_states**2 + n_states*n_symbols

        cls.range_start_probs = (0,n_states)
        cls.range_emission_probs = (n_states, (n_states + n_states*n_symbols))
        cls.range_transition_probs = ((n_states + n_states*n_symbols), cls.n_genes)

        # use slice_start_probs.start, slice_start_probs.stop for slice prob values
        cls.slice_start_probs = numpy.s_[0:n_states]
        cls.slice_emission_probs = numpy.s_[n_states: (n_states + n_states*n_symbols)]
        cls.slice_transition_probs = numpy.s_[(n_states + n_states*n_symbols): cls.n_genes]


        cls.emission_probs_legal_cutpoints = [i for i in range(
            cls.slice_emission_probs.start,
            cls.slice_emission_probs.stop,
            cls.n_symbols)]

        cls.transition_probs_legal_cutpoints = [i for i in range(
            cls.slice_transition_probs.start, 
            cls.slice_transition_probs.stop,
            cls.n_states)]

        cls.legal_cutpoints = [0] + cls.emission_probs_legal_cutpoints + cls.transition_probs_legal_cutpoints


    def to_hmm(self):
        emmission_probs_dicts = [dict(zip(self.alphabet, probs)) for probs in self.emission_probs_matrix]
        emmission_probs_dists = [DiscreteDistribution(probs) for probs in emmission_probs_dicts]
        
        t = self.transition_probs_matrix 
        e = emmission_probs_dists
        p = self.start_probs



        model = HiddenMarkovModel.from_matrix(
            self.transition_probs_matrix, 
            emmission_probs_dists, 
            self.start_probs
        )
        return model

    def normalize(self):
        for i in range(len(self.legal_cutpoints) -1):
            start = self.legal_cutpoints[i]
            stop = self.legal_cutpoints[i + 1]
            self.genes[start:stop] = normalize_vector(self.genes[start:stop])

        # self.start_probs = normalize_vector(self.start_probs)
        # self.transition_probs_matrix = normalize_matrix(self.transition_probs_matrix)
        # self.emission_probs_matrix = normalize_matrix(self.emission_probs_matrix)


    @property
    def start_probs(self):
        return self.genes[self.slice_start_probs]

    @start_probs.setter
    def start_probs(self, value: numpy.array):
        self.genes[self.slice_start_probs] = value

    # Transition Probs
    @property
    def transition_probs(self):
        return self.genes[self.slice_transition_probs]
    
    @transition_probs.setter
    def transition_probs(self, value: numpy.array):
        self.genes[self.slice_transition_probs] = value

    @property 
    def emission_probs(self):
        return self.genes[self.slice_emission_probs]
    
    @emission_probs.setter
    def emission_probs(self, value: numpy.array):
        self.genes[self.slice_emission_probs] = value


    @property
    def transition_probs_matrix(self):
        arr = self.genes[self.slice_transition_probs]
        shape = (self.n_states, self.n_states)
        return arr.reshape(shape)

    @transition_probs_matrix.setter
    def transition_probs_matrix(self, value: numpy.ndarray):
        arr = value.flatten()
        self.genes[self.slice_transition_probs] = arr
    
    # Emission Probs

    @property
    def emission_probs_matrix(self):
        arr = self.genes[self.slice_emission_probs]
        shape = (self.n_states, self.n_symbols)
        return arr.reshape(shape)

    @emission_probs_matrix.setter 
    def emission_probs_matrix(self, value: numpy.ndarray):
        arr = value.flatten()
        self.genes[self.slice_emission_probs] = arr









class GeneticAlgorithm: 
    def __init__(self, 
            initial_population: List[Individual], 
            n_generations: int, 
            fitness_func: Callable[[Individual, 'GeneticAlgorithm'], float], 
            parent_select_func: Callable[[List[Individual], int], List[List[Individual]]], 
            crossover_func: Callable[[List[Individual]], Individual], 
            mutation_func: Callable[[Individual], Individual], 
            keep_elitism=1,
            callbacks=[],
            normalize_after_mutation=True) -> None:
        """_summary_

        Args:
            initial_population (_type_): _description_
            n_generations (_type_): _description_
            fitness_func (_type_): sheeeesh boiiiiii
            parent_select_func (_type_): _description_
            crossover_func (_type_): _description_
            mutation_func (_type_): _description_
            keep_elitism (int, optional): _description_. Defaults to 1.
        """
        
        self.population = initial_population
        self.population_size = len(initial_population)
        self.n_generations = n_generations
        self.current_generation = 0
        self.fitness_func = fitness_func
        self.parent_select_func = parent_select_func
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.keep_elitism = keep_elitism
        self.callbacks = callbacks
        self.normalize_after_mutation = normalize_after_mutation

        # The number of pairs of parents per generation. (! A pair of parents can consist of more than two)
        self.parent_count = self.population_size - self.keep_elitism

        # Output
        self.result = {
            'min': [],
            'max': [],
            'mean': []
        }

        # keep track of the average fitness, min and max fitness
    
    def plot(self):

        # fig, ax = plt.subplots()  # Create a figure containing a single axes.

        x = range(self.n_generations)
        
        plt.plot(x, self.result['max'], label='max')
        plt.plot(x, self.result['mean'], label='mean')
        plt.plot(x, self.result['min'], label='min')

        plt.legend()
        plt.show()

        # ax.plot(range(self.n_generations), self.result['max']); 


    def start(self):
        # while self.current_generation < self.n_generations
        while self.current_generation < self.n_generations:

            # calculate fitness
            for individual in self.population:
                individual.fitness = self.fitness_func(individual, self)
                # track if fitness bigger than max or whatever

                
            # sort individuals by fitness
            self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True)

            # update result
            fitness_max = self.population[0].fitness
            fitness_min = self.population[-1].fitness
            fitness_sum = sum(x.fitness for x in self.population)
            fitness_mean = fitness_sum / self.population_size
            self.result['max'].append(fitness_max)
            self.result['min'].append(fitness_min)
            self.result['mean'].append(fitness_mean)


            # callbacks for custom functionality
            if self.callbacks:
                for cb in self.callbacks:
                    cb(self)       

            # children = population_size - keep_elitism
            # länge der parents muss = anzahl der offspring sein

            all_parents = self.parent_select_func(self.population, self.parent_count, self)
            if len(all_parents) != self.parent_count:
                raise Exception("The number of parents does not match the expected number")

            sooos = [self.crossover_func(parents, self) for parents in all_parents]
            # flatten offsprings
            offspring = [x for j in sooos for x in j]

            # mutate
            offspring = [self.mutation_func(x, self) for x in offspring]

            if self.normalize_after_mutation:
                for x in offspring:
                    x.normalize()

            # neue population ist keep elitism + offspring
            elites = self.population[:self.keep_elitism]
            self.population = elites + offspring
            

            # 5 paare von parents
            # jedes macht 2 kinder
            # 10 kinder


            # select parents for mating
            # > liste von partnern

            # crossover
            # for every pair of parents => crossover

            # children = summe aus allen crossover schmossovers


            # do mutation
            # self.population = map(self.mutation_func, self.population)
            # self.population = [self.mutation_func(x, self) for x in self.population]

            # neue generation builden:

            self.current_generation+=1
        return self.result