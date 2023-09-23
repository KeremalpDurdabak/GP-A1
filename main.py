from modules.BreedOperator import BreedOperator
from modules.Dataset import Dataset
from modules.OperatorSet import OperatorSet
from modules.Population import Population
from modules.Individual import Individual
from modules.ProblemDefinition import ProblemDefinition
from modules.RegisterList import RegisterList

def main(problem):

    population = Population(problem)
    population.create_population()

    breeder = BreedOperator(problem)
    breeder.compute_individuals_fitness(population.individuals)

    for gen in range(1, problem.gen_count):
        # Calculate and print the sum of fitness scores for the current generation
        total_fitness = sum(individual.fitnessScore for individual in population.individuals)
        print(f"Generation {gen}: Total Fitness = {total_fitness}")

        population.removePopulationGap()

        children = breeder.breed(population)
        breeder.compute_individuals_fitness(children)

        population.replacePopulationGap(children)


if __name__ == "__main__":

    # Declare Problem Parameters
    ############################

    # Dataset Path
    iris_dataset = Dataset("datasets/iris/iris.data")
    tictactoe_dataset_path = Dataset("datasets/tic+tac+toe+endgame/tic-tac-toe.data")

    # Number of Individuals in the Population
    population_count = 100

    # Max Instruction (Row) per each Individual
    max_instruction = 32

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
    mutation_prob = 0.1


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
