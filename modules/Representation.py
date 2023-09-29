from matplotlib import pyplot as plt
import numpy as np


class Representation:
    def __init__(self, population):
        self.population = population
        self.max_possible_fitness = self.population.problemDefinition.dataset.get_X().shape[0]


    def display_total_fitness(self):
        total_fitness = sum([individual.fitnessScore for individual in self.population.individuals])
        total_fitness_percentage = (total_fitness / (self.max_possible_fitness * len(self.population.individuals))) * 100
        print(f"Total Fitness: {total_fitness_percentage:.0f}%")

    def display_highest_fitness(self):
        highest_fitness = max([individual.fitnessScore for individual in self.population.individuals])
        highest_fitness_percentage = (highest_fitness / self.max_possible_fitness) * 100
        print(f"Highest Fitness: {highest_fitness_percentage:.0f}%")

    def display_all_fitness(self):
        sorted_fitness = sorted([individual.fitnessScore for individual in self.population.individuals], reverse=True)
        sorted_fitness_percentage = [f"{(x / self.max_possible_fitness * 100 ):.0f}%" for x in sorted_fitness]
        print(f"All Fitness Scores (Sorted): {sorted_fitness_percentage}")


    def display_highest_representation(self):
        highest_fitness_individual = max(self.population.individuals, key=lambda x: x.fitnessScore)
        representation = highest_fitness_individual.get_representation()
        print(f"Highest Individual's Representation: {representation}")


    def display_highest_fitness_class_prediction(self):
        highest_fitness_individual = max(self.population.individuals, key=lambda x: x.fitnessScore)
        class_predictions = highest_fitness_individual.get_predicted_classes()
        print(f"Highest Individual's Class Prediction: {class_predictions}")
        return class_predictions  # Return the class predictions

    def moving_average(self, data, window_size):
        """Calculate the moving average of a 1D numpy array."""
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


    def plot_fitness_scores(self, best_scores, mean_scores, worst_scores, best_class_predictions, window_size=10):
        plt.figure(figsize=(15, 6))

        # First graph for fitness scores
        plt.subplot(2, 1, 1)  # 2 row, 1 columns, first plot
        best_scores_percentage = [(x / self.max_possible_fitness) * 100 for x in best_scores]
        mean_scores_percentage = [(x / self.max_possible_fitness) * 100 for x in mean_scores]
        worst_scores_percentage = [(x / self.max_possible_fitness) * 100 for x in worst_scores]
        plt.plot(best_scores_percentage, label='Best Fitness', color='#3498DB')  # Blueish
        plt.plot(mean_scores_percentage, label='Mean Fitness', color='#F39C12')  # Yellow/Orange-ish
        plt.plot(worst_scores_percentage, label='Worst Fitness', color='#E74C3C')  # Redish
        plt.xlabel('Generation')
        plt.ylabel('Fitness Score (%)')
        plt.legend()

        # Second graph for class prediction percentages with highest predictions
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot

        best_class_predictions_np = np.array(best_class_predictions)
        highest_predictions = np.zeros(best_class_predictions_np.shape[1])

        for gen in range(best_class_predictions_np.shape[0]):
            for i in range(best_class_predictions_np.shape[1]):
                highest_predictions[i] = max(highest_predictions[i], best_class_predictions_np[gen, i])
            plt.plot(highest_predictions, marker='o', linestyle='-')

        plt.xlabel('Generation')
        plt.ylabel('Highest Class Prediction')
        plt.legend([f'Class {i+1}' for i in range(best_class_predictions_np.shape[1])])

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()
