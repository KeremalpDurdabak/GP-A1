import numpy as np
from modules.InstructionList import InstructionList
from modules.RegisterList import RegisterList  # Import the new class

class Individual:
    def __init__(self, problemDefinition):
        self.problemDefinition = problemDefinition
        self.instructionList = InstructionList(problemDefinition)
        self.fitnessScore = 0
        self.registerList = RegisterList()

    def create_individual(self):
        self.instructionList.generate_instruction_list()
        self.registerList.generate_register_list(self.problemDefinition.registerCount)

    def compute_individual_dataset_fitness_score(self):
        df_row_count = len(self.problemDefinition.dataset.get_df().index)
        for PC in range(df_row_count):
            self.instructionList.execute_instance(PC, self.registerList)
            self.compute_individual_instance_fitness_score(PC)
        return self.fitnessScore

    def compute_individual_instance_fitness_score(self, PC):
        instance_individual_label_verdict = self.registerList.argmax(self.problemDefinition.dataset.get_label_count())
        instance_real_label_verdict = self.problemDefinition.dataset.get_y().iloc[PC].values

        # Check if the individual's decision matches the real target label
        if np.array_equal(instance_individual_label_verdict, instance_real_label_verdict):
            self.fitnessScore += 1

    def get_representation(self):
        return self.instructionList.toString()
