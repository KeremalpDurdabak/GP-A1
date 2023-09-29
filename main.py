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
        
        # Train Generation
        population.removePopulationGap()
        children = breeder.breed(population)
        breeder.compute_individuals_fitness(children)
        population.replacePopulationGap(children)

        # Call Representation methods here
        representation.display_total_fitness()
        representation.display_highest_fitness()
        representation.display_all_fitness()
        representation.display_highest_representation()

        # Collect fitness scores for plotting
        all_fitness = [individual.fitnessScore for individual in population.individuals]
        best_scores.append(max(all_fitness))
        mean_scores.append(sum(all_fitness) / len(all_fitness))
        worst_scores.append(min(all_fitness))

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
    max_instruction = 24

    # Operators that will be used
    operators = OperatorSet(['+','-','*2','/2'])

    # Max number per each decode instruction (Source Select, Target Index, Source Index)
    # (Max number for the 'operator_select' is dynamically assumed by the OperatorSet class)
    max_decode_instructions = [2, 4, 9]

    # Number of Registers to use
    registerCount = 4

    # Percentage of worst fit Individuals to replace
    gap_percentage = 0.2

    # Generation Count
    gen_count = 100

    # Probability of a Mutation
    # 1. Probability of re-initializing an Instruction
    # 2. Probability of re-initializing an Instruction Bit
    # 3. Probability of randomly appending a new instruction to the child
    # 4. Probability of randomly removing an instruction from the child
    mutation_prob = [0.1, 0.3, 0.1, 0.05]  # List of 4 probabilities


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
