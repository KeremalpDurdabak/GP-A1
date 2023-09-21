class RegisterList:
    def __init__(self, register_count):
        self.register_count = register_count
        self.registers = []  # Array to store instances of the RegisterList class
        self.initialize_registers()

    def initialize_registers(self):
        # Initialize a registerList full of Registers
        for _ in range(self.register_count):  # Number of registers in the registerList
            new_register = 0
            self.registers.append(new_register)

    def argmax(self):
        max_reg = max(self.registers)
        argmax_reg = self.registers.index(max_reg)
        return argmax_reg

    def count(self):
        return len(self.registers)

    def reset_registers(self):
        # Reset all register values to 0
        for i in range(len(self.registers)):
            self.registers[i] = 0