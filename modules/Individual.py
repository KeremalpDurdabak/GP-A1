import numpy as np
from modules.InstructionList import InstructionList
from modules.RegisterList import RegisterList  # Import the new class

class Individual:
    def __init__(self, problemDefinition):
        self.problemDefinition = problemDefinition
        self.instructionList = InstructionList(problemDefinition)
        self.fitnessScore = 0
        self.registerList = RegisterList()

    def create_individual(self, isChild = False):
        if not isChild:
            self.instructionList.generate_instruction_list() 
        self.registerList.generate_register_list(self.problemDefinition.registerCount)

    def compute_individual_dataset_fitness_score(self):
        df_row_count = self.problemDefinition.dataset.get_X().shape[0]  # Use shape[0] to get the number of rows
        for PC in range(df_row_count):
            current_row = self.problemDefinition.dataset.get_X()[PC, :]  # Use NumPy slicing
            self.instructionList.execute_instance(current_row, self.registerList)
            self.compute_individual_instance_fitness_score(PC)
            self.registerList.reset_registers()
        return self.fitnessScore

    def compute_individual_instance_fitness_score(self, PC):
        instance_individual_label_verdict = self.registerList.argmax(self.problemDefinition.dataset.get_label_count())
        instance_real_label_verdict = self.problemDefinition.dataset.get_y()[PC, :]  # Use NumPy slicing

        # Check if the individual's decision matches the real target label
        if np.array_equal(instance_individual_label_verdict, instance_real_label_verdict):
            self.fitnessScore += 1

    def get_representation(self):
        return self.instructionList.toString()
    
    def get_predicted_classes(self):
        # Initialize an array to store the sum of correctly predicted classes
        predicted_classes_sum = np.zeros(self.problemDefinition.dataset.get_label_count())

        # Count the number of instances for each category in the y dataframe
        y_data = self.problemDefinition.dataset.get_y()
        total_instances_per_category = np.sum(y_data, axis=0)

        df_row_count = self.problemDefinition.dataset.get_X().shape[0]
        for PC in range(df_row_count):
            current_row = self.problemDefinition.dataset.get_X()[PC, :]
            self.instructionList.execute_instance(current_row, self.registerList)

            instance_individual_label_verdict = self.registerList.argmax(self.problemDefinition.dataset.get_label_count())
            instance_real_label_verdict = y_data[PC, :]

            # Check if the individual's decision matches the real target label
            if np.array_equal(instance_individual_label_verdict, instance_real_label_verdict):
                predicted_classes_sum += instance_real_label_verdict

            self.registerList.reset_registers()

        # Calculate the percentages
        predicted_classes_percentage = (predicted_classes_sum / total_instances_per_category) * 100

        return list(predicted_classes_percentage)


