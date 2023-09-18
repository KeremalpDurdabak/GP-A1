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

    def count(self):
        return len(self.registers)
