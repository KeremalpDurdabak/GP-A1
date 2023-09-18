from modules.variation.genetic_operators import crossover, mutate
from modules.selection.fitness_function import evaluate_fitness
from modules.selection.selection import breeder_selection
from modules.utils.utils import load_data, preprocess_data
from modules.population.population import initialize_population

def main(dataset_path):
    # Load and preprocess dataset
    data = load_data(dataset_path)
    data = preprocess_data(data)
    
    # Initialize population with 'programs'
    population = initialize_population(data)
    
    # Number of generations
    num_generations = 100
    
    for generation in range(num_generations):
        # Evaluate fitness of each individual in the population
        fitness_values = evaluate_fitness(population, data)
        
        # Breeder model for selection
        selected_individuals = breeder_selection(population, fitness_values)
        
        # Crossover and Mutation
        offspring = crossover(selected_individuals)
        offspring = mutate(offspring)
        
        # Update population for the next generation
        population = update_population(selected_individuals, offspring)
        
        # (Optional) Log or print current best individual and its fitness
        
    # Final reporting
    best_individual = find_best_individual(population)
    print(f"The best individual after {num_generations} generations is {best_individual}")

if __name__ == "__main__":

    # Adjust variables here
    iris_dataset = "datasets/iris/iris.data"
    tictactoe_dataset = "datasets/tic+tac+toe+endgame/tic-tac-toe.data"

    # Do GP!
    main(iris_dataset)
