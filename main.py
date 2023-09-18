from modules.OperatorSet import OperatorSet
from modules.Population import Population
from modules.Individual import Individual
from modules.ProblemDefinition import ProblemDefinition
from modules.RegisterList import RegisterList

def main(problem):

    # Initialize population with 'programs'
    population = Population(problem)


    for _ in 100:

        fitness = compute_fitness(population)

        if fitness >= 100:
            break

        parents = select_parents(fitness,population)
        child = apply_breeding(parents)
        child = apply_mutation(child)
        population = replace_population(child, population)

    print(population)

    

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
    registerList = RegisterList(4)

    ############################

    # Initialize Problem Class with the Problem Parameters
    iris_problem = ProblemDefinition(iris_dataset_path, population_count, registerList, max_instruction, operators, max_decode_instructions)
    tictactoe_problem = ProblemDefinition(tictactoe_dataset_path, population_count, registerList, max_instruction, operators, max_decode_instructions)

    # Do GP!
    main(iris_problem)
