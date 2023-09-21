from modules.InstructionList import InstructionList  # Import the new class

class Individual:
    def __init__(self, problemDefinition):
        self.problemDefinition = problemDefinition
        self.instructionList = InstructionList(self.problemDefinition)  # Use the new class
        self.argmaxList = []
        self.compute_instructions_through_dataset()
        print(self.argmaxList)

    def compute_instructions_through_dataset(self):
        for PC in range(self.problemDefinition.df.shape[0]):
            self.instructionList.compute_instructions_per_instance(PC)
            max_index = self.instructionList.get_argmax(self.problemDefinition.labelCount)
            self.argmaxList.append(max_index)
            self.instructionList.registerList.reset_registers()  # Reset registers here

    # def compute_fitness(self):

