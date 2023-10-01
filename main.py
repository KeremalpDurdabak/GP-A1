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

    highest_class_per_generation = []
    best_instruction_count = []

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

        highest_class_predictions = representation.display_highest_fitness_class_prediction()
        # Filter classes with scores above 50%
        filtered_classes = {k: v for k, v in highest_class_predictions.items() if v > 50}
        highest_class_per_generation.append(filtered_classes)

        highest_fitness_individual = max(population.individuals, key=lambda x: x.fitnessScore)
        best_instruction_count.append(len(highest_fitness_individual.instructionList.instructions))



        # Collect fitness scores for plotting
        all_fitness = [individual.fitnessScore for individual in population.individuals]
        best_scores.append(max(all_fitness))
        mean_scores.append(sum(all_fitness) / len(all_fitness))
        worst_scores.append(min(all_fitness))

    # Plot the fitness scores
    representation.plot_fitness_scores(best_scores, mean_scores, worst_scores, highest_class_per_generation, best_instruction_count)

    # Find the best individual from the training phase
    best_individual = max(population.individuals, key=lambda x: x.fitnessScore)
    best_individual.fitnessScore = 0

    # Update the dataset in the ProblemDefinition object
    problem.dataset.set_new_data(problem.dataset.X_test, problem.dataset.y_test)

    # Compute the fitness of the best individual on the test dataset
    test_score = breeder.compute_individuals_fitness([best_individual])

    # Convert the test score to a percentage
    test_score_percentage = (test_score / problem.dataset.X_test.shape[0]) * 100


    # Log the test score
    print(f"Test Score of the Best Individual: {test_score_percentage:.2f}%")
    print(f"Predicted Instances: {test_score}")





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
    mutation_prob = [0.1, 0.4, 0.1, 0.1]  # List of 4 probabilities


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
