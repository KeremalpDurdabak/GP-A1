import copy

import random
from modules.Individual import Individual
from modules.Instruction import Instruction
from modules.ProblemDefinition import ProblemDefinition

class BreedOperator:
    def __init__(self, population):
        self.population = population
        self.parents = []  # List of Individual objects
        self.children = []  # List of Individual objects
        self.currentChildTuple = []  # Initialize this attribute

    def generateParentPool(self):
        sorted_individuals = sorted(self.population.individuals, key=lambda ind: ind.fitnessScore, reverse=True)
        self.parents = sorted_individuals[:-self.population.problemDefinition.gap_num]  # Drop the worst 'gap_num' individuals
        return self.parents

    def generateChildPool(self):
        childrenCount = self.population.problemDefinition.gap_num  # This should be equal to gap_num
        self.children.clear()

        for _ in range(childrenCount // 2):
            self.createChildTuple()
            self.children.extend(self.currentChildTuple)
        return self.children

    def getParentTuple(self):
        # Extract the fitness scores of the parents
        fitness_scores = [parent.fitnessScore for parent in self.parents]
        
        # Use weighted random sampling to select two parents
        selected_parents = random.choices(self.parents, weights=fitness_scores, k=2)
        
        return selected_parents

    def createChildTuple(self):
        parentTuple = self.getParentTuple()
        self.crossover(parentTuple)
        self.mutation()

    def crossover(self, parentTuple):
        parent1 = parentTuple[0]
        parent2 = parentTuple[1]
        
        # Create new Individual instances for the offspring
        child1 = Individual(self.population.problemDefinition)
        child2 = Individual(self.population.problemDefinition)
        
        # Copy the instruction lists
        child1.instructionList.instructions = parent1.instructionList.instructions.copy()
        child2.instructionList.instructions = parent2.instructionList.instructions.copy()

        # Check if either parent has less than 2 instructions
        min_length = min(len(parent1.instructionList.instructions), len(parent2.instructionList.instructions))
        if min_length < 2:
            # Handle this special case (perhaps by not performing crossover)
            return

        # Randomly select a crossover point
        crossover_point = random.randint(1, min_length - 1)

        # Perform the crossover
        child1.instructionList.instructions[:crossover_point], child1.instructionList.instructions[crossover_point:] = \
            parent1.instructionList.instructions[:crossover_point], parent2.instructionList.instructions[crossover_point:]

        child2.instructionList.instructions[:crossover_point], child2.instructionList.instructions[crossover_point:] = \
            parent2.instructionList.instructions[:crossover_point], parent1.instructionList.instructions[crossover_point:]

        # Update the number of instructions for each child
        child1.instructionList.num_instructions = len(child1.instructionList.instructions)
        child2.instructionList.num_instructions = len(child2.instructionList.instructions)

        # Update the currentChildTuple with the new offspring
        self.currentChildTuple = [child1, child2]



    def mutation(self):
        if not self.currentChildTuple:  # Check if it's empty or None
            print("Warning: currentChildTuple is empty or not set.")
            return

        for child in self.currentChildTuple:
            if random.random() < self.population.problemDefinition.mutation_prob:
                # Decide the number of instructions to mutate, at most 3
                num_mutations = random.randint(1, min(3, len(child.instructionList.instructions)))
                
                # Randomly select 'num_mutations' unique instruction indices to mutate
                indices_to_mutate = random.sample(range(len(child.instructionList.instructions)), num_mutations)
                
                for instruction_index in indices_to_mutate:
                    # Create a new instruction
                    new_instruction = Instruction(self.population.problemDefinition)
                    
                    # Replace the old instruction with the new one
                    child.instructionList.instructions[instruction_index] = new_instruction


