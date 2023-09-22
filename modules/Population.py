from modules.Individual import Individual


class Population:
    def __init__(self, problemDefinition, next_generation=None):
        self.problemDefinition = problemDefinition
        self.new_individuals = []
        self.is_initial_population = True if next_generation is None else False
        self.individuals = next_generation if next_generation is not None else []
        if self.is_initial_population:
            self.initialize_population()

    def initialize_population(self):
        # Initialize a population full of Individuals
        for _ in range(self.problemDefinition.population_count):
            new_individual = Individual(self.problemDefinition)
            self.individuals.append(new_individual)

    def count(self):
        return len(self.individuals)

    def set_new_individuals(self, new_individuals):
        self.new_individuals = new_individuals

    def computeFitness(self):
        individuals_to_compute = self.new_individuals if not self.is_initial_population else self.individuals
        for individual in individuals_to_compute:
            individual.compute_fitness()
        self.new_individuals.clear()
