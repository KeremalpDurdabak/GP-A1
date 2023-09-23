import numpy as np

class RegisterList:
    def __init__(self):
        self.registers = []  # Array to store instances of the RegisterList class

    def generate_register_list(self, registerCount):
        # Initialize a registerList full of Registers
        for _ in range(registerCount):  # Number of registers in the registerList
            new_register = 0
            self.registers.append(new_register)

    def argmax(self, labelCount):
        if labelCount > len(self.registers):
            print(f'Label Count: {labelCount}')
            print(f'Register Count: {len(self.registers)}')
            raise ValueError("labelCount cannot be greater than the number of registers.")
        
        # Find the index of the maximum value among the first labelCount elements
        max_index = np.argmax(self.registers[:labelCount])
        
        # Use np.eye to generate an identity matrix and select the row at max_index
        result_array = np.eye(labelCount, dtype=int)[max_index]
        
        return result_array

    def count(self):
        return len(self.registers)

    def reset_registers(self):
        # Reset all register values to 0
        for i in range(len(self.registers)):
            self.registers[i] = 0
