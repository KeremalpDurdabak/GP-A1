from modules.BreedOperator import BreedOperator
from modules.Dataset import Dataset
from modules.OperatorSet import OperatorSet
from modules.Population import Population
from modules.Individual import Individual
from modules.ProblemDefinition import ProblemDefinition
from modules.RegisterList import RegisterList
from modules.Representation import Representation  # Import the Representation class

def main(problem):
    population = Population(problem)
    population.create_population()

    breeder = BreedOperator(problem)
    breeder.compute_individuals_fitness(population.individuals)

    # Initialize Representation class
    representation = Representation(population)


    # Initialize lists to store fitness scores
    best_scores = []
    mean_scores = []
    worst_scores = []

    for gen in range(1, problem.gen_count):
        print(f"Generation {gen}:")
        
        # Call Representation methods
        representation.display_total_fitness()
        representation.display_highest_fitness()
        representation.display_all_fitness()
        representation.display_highest_representation()

        # Collect fitness scores
        all_fitness = [individual.fitnessScore for individual in population.individuals]
        max_fitness = max(all_fitness)
        
        max_fitness_percentage = (max_fitness / representation.max_possible_fitness) * 100 * 100  # Aligned with Representation class

        print(max_fitness_percentage)
        # Terminate if any individual's fitness exceeds 85%
        # if max_fitness_percentage >= 90:  # Adjusted to align with Representation class
        #     print("Stopping criterion reached.")
        #     break

        # Train Generation
        population.removePopulationGap()

        # Calculate the worst and mean score among the remaining individuals
        worst_among_remaining = min(individual.fitnessScore for individual in population.individuals)
        mean_among_remaining = sum(individual.fitnessScore for individual in population.individuals) / len(population.individuals)

        # Append to lists for plotting
        best_scores.append(max_fitness)
        mean_scores.append(mean_among_remaining)  # Use mean_among_remaining here
        worst_scores.append(worst_among_remaining)  # Use worst_among_remaining here
        
        children = breeder.breed(population)
        breeder.compute_individuals_fitness(children)
        population.replacePopulationGap(children)


    # Plot the fitness scores
    representation.plot_fitness_scores(best_scores, mean_scores, worst_scores)



if __name__ == "__main__":

    # Declare Problem Parameters
    ############################

    # Dataset Path
    iris_dataset = Dataset("datasets/iris/iris.data")
    tictactoe_dataset = Dataset("datasets/tic+tac+toe+endgame/tic-tac-toe.data")

    # Number of Individuals in the Population
    population_count = 100

    # Max Instruction (Row) per each Individual
    max_instruction = 16

    # Operators that will be used
    operators = OperatorSet(['+','-','*2','/2'])

    # Max number per each decode instruction (Source Select, Target Index, Source Index)
    # (Max number for the 'operator_select' is dynamically assumed by the OperatorSet class)
    max_decode_instructions = [2, 4, 4]

    # Number of Registers to use
    registerCount = 4

    # Percentage of worst fit Individuals to replace
    gap_percentage = 0.2

    # Generation Count
    gen_count = 1000

    # Probability of a Mutation
    # 1. Probability of re-initializing an Instruction
    # 2. Probability of re-initializing an Instruction Bit
    # 3. Probability of randomly appending a new instruction to the child
    # 4. Probability of randomly removing an instruction from the child
    mutation_prob = [0.1, 0.3, 0.05, 0.05]  # List of 4 probabilities


    ############################

    # Initialize Problem Class with the Problem Parameters
    problem = ProblemDefinition(iris_dataset, 
                                gen_count, 
                                population_count,
                                gap_percentage, 
                                registerCount, 
                                max_instruction, 
                                operators, 
                                max_decode_instructions,
                                mutation_prob
                                )
    
    # Do GP!
    main(problem)
