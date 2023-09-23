import random
from modules.Instruction import Instruction
from modules.RegisterList import RegisterList
import numpy as np

class InstructionList:
    def __init__(self, problemDefinition):
        self.problemDefinition = problemDefinition
        self.instructions = []
        self.num_instructions = 0
    
    def generate_instruction_list(self):
        self.num_instructions = random.randint(1, self.problemDefinition.max_instruction)
        for _ in range(self.num_instructions):
            new_instruction = Instruction(self.problemDefinition)
            new_instruction.generate_instruction()
            self.instructions.append(new_instruction)

    def execute_instance(self, PC, registerList):
        for i in range(self.num_instructions):
            self.instructions[i].execute_instruction(PC, registerList)

    def toString(self):
        instruction_string_list = []
        for i in range(self.num_instructions):
            instruction_string_list.append(i.toString())
        return instruction_string_list