from matplotlib import pyplot as plt
import numpy as np
import matplotlib.colors as mcolors



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
        class_prediction_dict = {f"Class {i+1}": score for i, score in enumerate(class_predictions)}
        print(f"Highest Individual's Class Prediction: {class_prediction_dict}")
        return class_prediction_dict  # Return the class predictions as a dictionary



    def plot_fitness_scores(self, best_scores, mean_scores, worst_scores, highest_class_per_generation):
        plt.figure(figsize=(8,6))

        # First graph for fitness scores
        plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
        best_scores_percentage = [(x / self.max_possible_fitness) * 100 for x in best_scores]
        mean_scores_percentage = [(x / self.max_possible_fitness) * 100 for x in mean_scores]
        worst_scores_percentage = [(x / self.max_possible_fitness) * 100 for x in worst_scores]
        plt.plot(best_scores_percentage, label='Best Fitness', color='#3498DB')  # Blueish
        plt.plot(mean_scores_percentage, label='Mean Fitness', color='#F39C12')  # Yellow/Orange-ish
        plt.plot(worst_scores_percentage, label='Worst Fitness', color='#E74C3C')  # Redish
        plt.xlabel('Generation')
        plt.ylabel('Fitness Score (%)')
        plt.legend()

        # Add a new subplot for the highest class per generation
        plt.subplot(2, 1, 2)  # 2 rows, 1 column, second plot

        # Get unique class labels
        unique_class_labels = list({f"Class {i+1}" for i in range(self.population.problemDefinition.dataset.get_label_count())})

        # Create a color map for the classes
        colors = list(mcolors.TABLEAU_COLORS.values())
        class_to_color = {class_label: colors[i % len(colors)] for i, class_label in enumerate(unique_class_labels)}

        # Plot the dots with distinct colors for each class
        for i, highest_classes in enumerate(highest_class_per_generation):
            for class_label in highest_classes:
                y_index = unique_class_labels.index(class_label)
                plt.scatter(i, y_index, marker='o', color=class_to_color[class_label], s=15)

        plt.yticks(range(len(unique_class_labels)), unique_class_labels)
        plt.xlabel('Generation')
        plt.ylabel('Highest Scoring Class')
        plt.tight_layout()  # Adjust layout to fit plots better
        plt.show()

