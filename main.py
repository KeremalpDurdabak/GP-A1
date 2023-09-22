from modules.BreedOperator import BreedOperator
from modules.OperatorSet import OperatorSet
from modules.Population import Population
from modules.Individual import Individual
from modules.ProblemDefinition import ProblemDefinition
from modules.RegisterList import RegisterList

def main(problem):
    population = Population(problem)

    for gen in range(1, problem.gen_count):
        population.computeFitness()
        breeder = BreedOperator(population)

        # Calculate and print the sum of fitness scores for the current generation
        total_fitness = sum(individual.fitnessScore for individual in population.individuals)
        print(f"Generation {gen}: Total Fitness = {total_fitness}")

        parents = breeder.generateParentPool()
        children = breeder.generateChildPool()

        # Combine parents and children to form the next generation
        next_gen = parents + children  # Simply concatenate the lists

        # Initialize a new population with individuals from the previous generation
        population = Population(problem, next_generation=next_gen)
        population.set_new_individuals(children)  # Set new individuals for the next generation


if __name__ == "__main__":

    # Declare Problem Parameters
    ############################

    # Dataset Path
    iris_dataset_path = "datasets/iris/iris.data"
    tictactoe_dataset_path = "datasets/tic+tac+toe+endgame/tic-tac-toe.data"

    # Number of Individuals in the Population
    population_count = 100

    # Max Instruction (Row) per each Individual
    max_instruction = 63

    # Operators that will be used
    operators = OperatorSet(['+','-','*2','/2'])

    # Max number per each decode instruction (Source Select, Target Index, Source Index)
    # (Max number for the 'operator_select' is dynamically assumed by the OperatorSet class)
    max_decode_instructions = [2, 4, 4]

    # Number of Registers to use
    registerCount = 4

    # Number of Categorical Labels to predict
    labelCount = 3

    # Percentage of worst fit Individuals to replace
    gap_percentage = 0.2

    # Generation Count
    gen_count = 1000

    # Probability of a Mutation
    mutation_prob = 0.2


    ############################

    # Initialize Problem Class with the Problem Parameters
    problem = ProblemDefinition(iris_dataset_path, 
                                gen_count, 
                                gap_percentage, 
                                population_count, 
                                registerCount, 
                                labelCount, 
                                max_instruction, 
                                operators, 
                                max_decode_instructions,
                                mutation_prob
                                )
    
    # Do GP!
    main(problem)
