import numpy

def single_row_cutpoint(cut_points):
    # cut_points = get_cut_points(num_states, num_symbols)

    def crossover_func(parents, offspring_size, ga_instance):
        num_offspring = offspring_size[0]
        offspring = []
        idx = 0

        while len(offspring) < num_offspring:
            parent1 = parents[idx % parents.shape[0], :].copy()
            parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

            random_split_point = numpy.random.choice(cut_points)

            parent1[random_split_point:] = parent2[random_split_point:]

            offspring.append(parent1)

            idx += 1

        return numpy.array(offspring)

    return crossover_func

