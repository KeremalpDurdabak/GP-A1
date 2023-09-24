class Representation:
    def __init__(self, population):
        self.population = population
        self.max_possible_fitness = len(self.population.problemDefinition.dataset.get_df().index) * len(self.population.individuals)

    def display_total_fitness(self):
        total_fitness = sum(individual.fitnessScore for individual in self.population.individuals)
        total_fitness_percentage = (total_fitness / self.max_possible_fitness) * 100
        print(f"Total Fitness: {round(total_fitness_percentage):.0f}%")

    def display_highest_fitness(self):
        highest_fitness = max(individual.fitnessScore for individual in self.population.individuals)
        highest_fitness_percentage = (highest_fitness / self.max_possible_fitness) * 100 * 100
        print(f"Highest Fitness: {round(highest_fitness_percentage):.0f}%")

    def display_all_fitness(self):
        all_fitness = [individual.fitnessScore for individual in self.population.individuals]
        sorted_fitness = sorted(all_fitness, reverse=True)
        sorted_fitness_percentage = [f"{round((x / self.max_possible_fitness) * 100 * 100):.0f}%" for x in sorted_fitness]
        print(f"All Fitness Scores (Sorted): {sorted_fitness_percentage}")


    def display_highest_representation(self):
        highest_fitness_individual = max(self.population.individuals, key=lambda x: x.fitnessScore)
        representation = highest_fitness_individual.get_representation()
        print(f"Highest Individual's Representation: {representation}")

    def plot_fitness_scores(self, best_scores, mean_scores, worst_scores):
        plt.figure(figsize=(10, 6))
        plt.plot(best_scores, label='Best Fitness')
        plt.plot(mean_scores, label='Mean Fitness')
        plt.plot(worst_scores, label='Worst Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Score')
        plt.legend()
        plt.show()
