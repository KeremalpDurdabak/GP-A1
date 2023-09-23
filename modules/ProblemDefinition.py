import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class ProblemDefinition:
    def __init__(self, dataset, gen_count, population_count, gap_percentage, registerCount, max_instruction, operators, max_decode_instructions, mutation_prob):
        self.dataset = dataset
        self.gen_count = gen_count
        self.population_count = population_count
        self.gap_num = int(self.population_count * gap_percentage)
        self.registerCount = registerCount
        self.max_instruction = max_instruction
        self.operators = operators
        self.max_decode_instructions = max_decode_instructions
        self.mutation_prob = mutation_prob