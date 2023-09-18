import random
from modules.Instruction import Instruction

class Individual:
    def __init__(self, problemDefinition):
        self.problemDefinition = problemDefinition
        self.instructions = []  # List to store instances of the Instruction class
        self.initialize_instructions()

    def initialize_instructions(self):
        # Determine the random number of instructions for this individual
        num_instructions = random.randint(1, self.problemDefinition.max_instruction)
        
        # Initialize the instructions
        for PC in range(num_instructions):
            new_instruction = Instruction(self.problemDefinition, PC)
            self.instructions.append(new_instruction)
