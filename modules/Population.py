from modules.Individual import Individual

class Population:
    def __init__(self, problemDefinition):
        self.problemDefinition = problemDefinition
        self.individuals = []  # Array to store instances of the Individual class
        self.initialize_population()

    def initialize_population(self):
        # Initialize a population full of Individuals
        for _ in range(self.problemDefinition.population_count):  # Number of individuals in the population
            new_individual = Individual(self.problemDefinition)
            self.individuals.append(new_individual)

    def count(self):
        return len(self.individuals)
