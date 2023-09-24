class Representation:
    def __init__(self, population):
        self.population = population

    def display_total_fitness(self):
        total_fitness = sum(individual.fitnessScore for individual in self.population.individuals)
        print(f"Total Fitness: {total_fitness}")

    def display_highest_fitness(self):
        highest_fitness = max(individual.fitnessScore for individual in self.population.individuals)
        print(f"Highest Fitness: {highest_fitness}")

    def display_all_fitness(self):
        all_fitness = [individual.fitnessScore for individual in self.population.individuals]
        sorted_fitness = sorted(all_fitness, reverse=True)
        print(f"All Fitness Scores (Sorted): {sorted_fitness}")

    def display_highest_representation(self):
        highest_fitness_individual = max(self.population.individuals, key=lambda x: x.fitnessScore)
        representation = highest_fitness_individual.get_representation()
        print(f"Highest Individual's Representation: {representation}")