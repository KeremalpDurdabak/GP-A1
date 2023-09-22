import random
from modules.RegisterList import RegisterList

class Instruction:
    def __init__(self, problemDefinition):
        self.problemDefinition = problemDefinition

        self.source_select = random.randint(0, self.problemDefinition.max_decode_instructions[0])
        self.target_index = random.randint(0, self.problemDefinition.max_decode_instructions[1])
        self.operator_select = random.randint(0, len(self.problemDefinition.operators.operators) - 1)
        self.source_index = random.randint(0, self.problemDefinition.max_decode_instructions[2])

    def compute_instruction(self,PC,registerList):
        if self.source_select == 0:
            # Pull the value from the dataset's feature dataframe
            # Use modulus to wrap the index
            source_value = self.problemDefinition.df.iloc[PC, self.source_index % self.problemDefinition.df.shape[1]]
        else:
            # Use the register
            source_value = registerList.registers[self.source_index % registerList.count()]

        # Compute the instruction
        target_idx_mod = self.target_index % registerList.count()
        registerList.registers[target_idx_mod] = self.problemDefinition.operators.compute(
            self.operator_select,
            registerList.registers[target_idx_mod],
            source_value
        )

    def __repr__(self):
        return f"Instruction(source_select={self.source_select}, target_index={self.target_index}, operator_select={self.operator_select}, source_index={self.source_index})"
