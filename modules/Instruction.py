import random
from modules.OperatorSet import OperatorSet

class Instruction:
    def __init__(self, problemDefinition):
        self.problemDefinition = problemDefinition
        self.source_select = -1
        self.target_index = -1
        self.operator_select = -1
        self.source_index = -1

    def generate_instruction(self):
        self.source_select = random.randint(0, self.problemDefinition.max_decode_instructions[0] - 1)
        self.target_index = random.randint(0, self.problemDefinition.max_decode_instructions[1] - 1)
        self.operator_select = random.randint(0, len(self.problemDefinition.operators.operators) - 1)
        self.source_index = random.randint(0, self.problemDefinition.max_decode_instructions[2] - 1)

    def mutate_instruction_bits(self):
        # Randomly decide which bits to mutate
        bits_to_mutate = random.sample(['source_select', 'target_index', 'operator_select', 'source_index'], random.randint(0, 3))

        # Mutate the selected bits
        if 'source_select' in bits_to_mutate:
            self.source_select = random.randint(0, self.problemDefinition.max_decode_instructions[0] - 1)
        if 'target_index' in bits_to_mutate:
            self.target_index = random.randint(0, self.problemDefinition.max_decode_instructions[1] - 1)
        if 'operator_select' in bits_to_mutate:
            self.operator_select = random.randint(0, len(self.problemDefinition.operators.operators) - 1)
        if 'source_index' in bits_to_mutate:
            self.source_index = random.randint(0, self.problemDefinition.max_decode_instructions[2] - 1)

    def execute_instruction(self, PC, registerList):
        if self.source_select == 0:
            # Pull the value from the dataset's feature dataframe
            # Use modulus to wrap the index
            # % self.problemDefinition.dataset.get_X().shape[1]
            source_value = self.problemDefinition.dataset.get_X().iloc[PC, self.source_index % self.problemDefinition.dataset.get_X().shape[1]]#!
        else:
            # Use the register
            source_value = registerList.registers[self.source_index % registerList.count()]

        # Compute the instruction
        registerList.registers[self.target_index % registerList.count()] = self.problemDefinition.operators.compute(
            self.operator_select,
            registerList.registers[self.target_index % registerList.count()],
            source_value
        )

    def toString(self):
        op_string = OperatorSet.represent(self.operator_select)
        target_part = f'R[{self.target_index}] <- R[{self.target_index}]'
        op_part = f' {op_string} '
        source_part = f'X[{self.source_index}]' if self.source_select == 0 else f'R[{self.source_index}]'
        target_op = target_part + op_part
        return target_op if op_string not in ['+', '-'] else (target_op + source_part)