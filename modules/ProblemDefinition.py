import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class ProblemDefinition:
    def __init__(self, dataset_path, population_count, registerCount, labelCount, max_instruction, operators, max_decode_instructions):
        self.population_count = population_count
        self.registerCount = registerCount
        self.labelCount = labelCount
        self.dataset_path = dataset_path
        self.max_instruction = max_instruction
        self.operators = operators
        self.max_decode_instructions = max_decode_instructions
        self.df = self.load_data()
        self.df = self.one_hot_encode_target()

    def load_data(self):
        data = pd.read_csv(self.dataset_path, header=None)  # Assuming the dataset doesn't have a header
        return data
    
    def one_hot_encode_target(self):
        if self.df is None:
            print("Data not loaded. Please load the data first.")
            return None

        # Separate features and target
        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]

        # One-hot encode the target variable
        encoder = OneHotEncoder(sparse=False)
        y_onehot = encoder.fit_transform(y.values.reshape(-1, 1))

        # Convert one-hot encoded array back to DataFrame for easier manipulation
        y_onehot_df = pd.DataFrame(y_onehot, columns=encoder.get_feature_names_out(input_features=['target']))

        # Concatenate features and one-hot encoded target
        preprocessed_data = pd.concat([X, y_onehot_df], axis=1)
        
        return preprocessed_data