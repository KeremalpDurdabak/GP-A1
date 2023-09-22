from modules.InstructionList import InstructionList  # Import the new class

class Individual:
    def __init__(self, problemDefinition, instructionList=None):
        self.problemDefinition = problemDefinition
        if instructionList:
            self.instructionList = instructionList
        else:
            self.instructionList = InstructionList(self.problemDefinition)
        self.argmaxList = []
        self.fitnessScore = 0
        self.compute_instructions_through_dataset()


    def compute_instructions_through_dataset(self):
        for PC in range(self.problemDefinition.df.shape[0]):
            self.instructionList.compute_instructions_per_instance(PC)
            max_index = self.instructionList.get_argmax(self.problemDefinition.labelCount)
            self.argmaxList.append(max_index)
            self.instructionList.registerList.reset_registers()  # Reset registers here

    def compute_fitness(self):
        label_columns = self.problemDefinition.df.columns[-self.problemDefinition.labelCount:]  # Get the last 'labelCount' columns
        
        for i in range(len(self.argmaxList)):
            predicted_label = self.argmaxList[i]
            actual_label_vector = self.problemDefinition.df.iloc[i][label_columns].tolist()
            
            # Find the index of the actual label (where the value is 1 in one-hot encoding)
            actual_label = actual_label_vector.index(1)
            
            # Compare and update the fitness score
            if predicted_label == actual_label:
                self.fitnessScore += 1
