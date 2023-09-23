import random

from modules.Individual import Individual
from modules.Instruction import Instruction
import numpy as np


class BreedOperator:
    def __init__(self, problemDefinition):
        self.problemDefinition = problemDefinition

    def compute_individuals_fitness(self, individuals):
        individuals_fitness = 0
        for i in individuals:
            individuals_fitness += i.compute_individual_dataset_fitness_score()
        return individuals_fitness

    def select_parents(self, population):
        weights = [individual.fitnessScore for individual in population.individuals]
        selected_indices = np.random.choice(len(population.individuals), 2, replace=False, p=np.array(weights)/sum(weights))
        selected_parents = [population.individuals[i] for i in selected_indices]
        return selected_parents
    

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(0, min(len(parent1.instructionList.instructions), len(parent2.instructionList.instructions)) - 1)
        
        child1_instructions = parent1.instructionList.instructions[:crossover_point] + parent2.instructionList.instructions[crossover_point:]
        child2_instructions = parent2.instructionList.instructions[:crossover_point] + parent1.instructionList.instructions[crossover_point:]
        
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
            mutation_point = random.randint(0, len(individual.instructionList.instructions) - 1)
            new_instruction = Instruction(self.problemDefinition)
            new_instruction.generate_instruction()
            individual.instructionList.instructions[mutation_point] = new_instruction

    def breed(self, population):
        children = []
        for _ in range(self.problemDefinition.gap_num // 2):
            parent1, parent2 = self.select_parents(population)
            
            child1, child2 = self.crossover(parent1, parent2)
            
            self.mutate(child1)
            self.mutate(child2)
            
            children.extend([child1, child2])
        
        return children
