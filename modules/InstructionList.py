import random
from modules.Instruction import Instruction
from modules.RegisterList import RegisterList

class InstructionList:
    def __init__(self, problemDefinition):
        self.problemDefinition = problemDefinition
        self.registerList = RegisterList(self.problemDefinition.registerCount)
        self.instructions = []
        self.num_instructions = random.randint(1, self.problemDefinition.max_instruction)
        self.initialize_instructions()

    def initialize_instructions(self):
        for _ in range(self.num_instructions):
            instruction = Instruction(self.problemDefinition)
            self.instructions.append(instruction)

    def compute_instructions_per_instance(self, PC):
        for i in range(self.num_instructions):
            self.instructions[i].compute_instruction(PC, self.registerList)

    def get_argmax(self, labelCount):
        sliced_registerList = self.registerList.registers[:labelCount]
        max_val = max(sliced_registerList)
        max_index = sliced_registerList.index(max_val)
        return max_index