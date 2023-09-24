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

    # def select_parents(self, population):
    #     weights = [individual.fitnessScore for individual in population.individuals]
    #     selected_indices = np.random.choice(len(population.individuals), 2, replace=False, p=np.array(weights)/sum(weights))
    #     selected_parents = [population.individuals[i] for i in selected_indices]
    #     return selected_parents
    
    def select_parents(self, population):
        return random.sample(population.individuals, 2)

    def crossover(self, parent1, parent2):
        min_length = min(len(parent1.instructionList.instructions), len(parent2.instructionList.instructions))
        
        point1, point2 = sorted(random.sample(range(min_length), 2))
        
        child1_instructions = (
            parent1.instructionList.instructions[:point1] +
            parent2.instructionList.instructions[point1:point2] +
            parent1.instructionList.instructions[point2:]
        )
        
        child2_instructions = (
            parent2.instructionList.instructions[:point1] +
            parent1.instructionList.instructions[point1:point2] +
            parent2.instructionList.instructions[point2:]
        )
        
        child1 = Individual(self.problemDefinition)
        child1.create_individual(isChild=True)
        child2 = Individual(self.problemDefinition)
        child2.create_individual(isChild=True)
        
        child1.instructionList.instructions = child1_instructions
        child2.instructionList.instructions = child2_instructions

        child1.instructionList.num_instructions = len(child1_instructions)
        child2.instructionList.num_instructions = len(child2_instructions)
        
        return child1, child2

    def mutate(self, individual):
        if random.random() < self.problemDefinition.mutation_prob:
            # Randomly decide the number of mutations (up to 3)
            num_mutations = random.randint(1, 4 - 1)
            
            # Mutate existing instructions
            for _ in range(num_mutations):
                mutation_point = random.randint(0, len(individual.instructionList.instructions) - 1)
                new_instruction = Instruction(self.problemDefinition)
                new_instruction.generate_instruction()
                individual.instructionList.instructions[mutation_point] = new_instruction

            # 30% chance to add a new instruction, if it doesn't exceed max_instruction
            if random.random() < 0.3 and len(individual.instructionList.instructions) < self.problemDefinition.max_instruction:
                new_instruction = Instruction(self.problemDefinition)
                new_instruction.generate_instruction()
                individual.instructionList.instructions.append(new_instruction)
                individual.instructionList.num_instructions += 1

            # 20% chance to remove an instruction, if there are more than 2
            if random.random() < 0.2 and len(individual.instructionList.instructions) > 2:
                remove_point = random.randint(0, len(individual.instructionList.instructions) - 1)
                del individual.instructionList.instructions[remove_point]
                individual.instructionList.num_instructions -= 1

            # 20% chance for bit-level mutation
            if random.random() < 0.3:
                mutation_point = random.randint(0, len(individual.instructionList.instructions) - 1)
                individual.instructionList.instructions[mutation_point].mutate_instruction_bits()

    # def mutate(self, individual):
    #     if random.random() < self.problemDefinition.mutation_prob:
    #         mutation_point = random.randint(0, len(individual.instructionList.instructions) - 1)
    #         individual.instructionList.instructions[mutation_point].mutate_instruction_bits()


    def breed(self, population):
        children = []
        for _ in range(self.problemDefinition.gap_num // 2):
            parent1, parent2 = self.select_parents(population)
            
            child1, child2 = self.crossover(parent1, parent2)
            
            self.mutate(child1)
            self.mutate(child2)
            
            children.extend([child1, child2])
        
        return children
