import random
import numpy as np
from modules.Individual import Individual
from modules.Instruction import Instruction


class BreedOperator:
    def __init__(self, problemDefinition):
        self.problemDefinition = problemDefinition

    def compute_individuals_fitness(self, individuals):
        individuals_fitness = 0
        for i in individuals:
            individuals_fitness += i.compute_individual_dataset_fitness_score()
        return individuals_fitness

    def select_parents(self, population, type='agnostic'):
        if type == 'elitist':
            weights = [individual.fitnessScore for individual in population.individuals]
            selected_indices = np.random.choice(len(population.individuals), 2, replace=False, p=np.array(weights)/sum(weights))
            selected_parents = [population.individuals[i] for i in selected_indices]
            
        elif type == 'agnostic':
            selected_parents = random.sample(population.individuals, 2)
            
        return selected_parents

    def crossover(self, parent1, parent2, type='single'):
        child1 = Individual(self.problemDefinition)
        child1.create_individual(isChild=True)
        child2 = Individual(self.problemDefinition)
        child2.create_individual(isChild=True)
        
        if type == 'single':
            min_length = min(len(parent1.instructionList.instructions), len(parent2.instructionList.instructions))
            point = random.randint(0, min_length - 1)
            child1_instructions = parent1.instructionList.instructions[:point] + parent2.instructionList.instructions[point:]
            child2_instructions = parent2.instructionList.instructions[:point] + parent1.instructionList.instructions[point:]

        elif type == 'double':
            min_length = min(len(parent1.instructionList.instructions), len(parent2.instructionList.instructions))
            point1, point2 = sorted(random.sample(range(min_length), 2))
            child1_instructions = parent1.instructionList.instructions[:point1] + parent2.instructionList.instructions[point1:point2] + parent1.instructionList.instructions[point2:]
            child2_instructions = parent2.instructionList.instructions[:point1] + parent1.instructionList.instructions[point1:point2] + parent2.instructionList.instructions[point2:]

        elif type == 'multi':
            parent1_instructions = parent1.instructionList.instructions
            parent2_instructions = parent2.instructionList.instructions
            child1_instructions = parent1_instructions.copy()
            child2_instructions = parent2_instructions.copy()

            # Randomly select indices for instructions to swap
            swap_indices1 = random.sample(range(len(parent1_instructions)), random.randint(1, len(parent1_instructions)))
            swap_indices2 = random.sample(range(len(parent2_instructions)), random.randint(1, len(parent2_instructions)))

            # Swap the selected instructions
            for i1, i2 in zip(swap_indices1, swap_indices2):
                if i1 < len(child1_instructions) and i2 < len(child2_instructions):
                    child1_instructions[i1], child2_instructions[i2] = child2_instructions[i2], child1_instructions[i1]

            # Assign the new instructions to the children and update the number of instructions
            child1.instructionList.instructions = child1_instructions
            child2.instructionList.instructions = child2_instructions
            child1.instructionList.num_instructions = len(child1_instructions)
            child2.instructionList.num_instructions = len(child2_instructions)



        # Assign the new instructions to the children and update the number of instructions
        child1.instructionList.instructions = child1_instructions
        child2.instructionList.instructions = child2_instructions
        child1.instructionList.num_instructions = len(child1_instructions)
        child2.instructionList.num_instructions = len(child2_instructions)
        
        return child1, child2

    def mutate_instruction(self, child):
        mutation_point = random.randint(0, len(child.instructionList.instructions) - 1)
        new_instruction = Instruction(self.problemDefinition)
        new_instruction.generate_instruction()
        child.instructionList.instructions[mutation_point] = new_instruction

    def mutate_instruction_bit(self, child):
        mutation_point = random.randint(0, len(child.instructionList.instructions) - 1)
        child.instructionList.instructions[mutation_point].mutate_instruction_bits()

    def mutate_add_instruction(self, child):
        if len(child.instructionList.instructions) < self.problemDefinition.max_instruction:
            new_instruction = Instruction(self.problemDefinition)
            new_instruction.generate_instruction()
            child.instructionList.instructions.append(new_instruction)
            child.instructionList.num_instructions += 1

    def mutate_remove_instruction(self, child):
        if len(child.instructionList.instructions) > 2:
            remove_point = random.randint(0, len(child.instructionList.instructions) - 1)
            del child.instructionList.instructions[remove_point]
            child.instructionList.num_instructions -= 1

    def mutate(self, child):
        # Mapping of string types to methods
        mutation_methods = [
            self.mutate_instruction,
            self.mutate_instruction_bit,
            self.mutate_add_instruction,
            self.mutate_remove_instruction
        ]
        
        # Apply the mutations based on their probabilities
        for i, method in enumerate(mutation_methods):
            if random.random() < self.problemDefinition.mutation_prob[i]:
                method(child)


    def breed(self, population):
        children = []
        for _ in range(self.problemDefinition.gap_num // 2):
            parent1, parent2 = self.select_parents(population, 'agnostic')
            
            child1, child2 = self.crossover(parent1, parent2, 'double')
            
            self.mutate(child1)
            self.mutate(child2)
            
            children.extend([child1, child2])
        
        return children
