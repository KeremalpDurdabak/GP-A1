import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class Dataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = self.load_data()
        self.label_count = 1  # Initialize with 1 assuming only one target column before encoding

        self.preprocess_data()

    def preprocess_data(self):
        self.df = self.one_hot_encode_target()
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def get_df(self):
        return self.df
    
    def get_label_count(self):
        return self.label_count

    def load_data(self):
        data = pd.read_csv(self.dataset_path, header=None)  # Assuming the dataset doesn't have a header
        return data
    
    def get_X(self):
        return self.df.iloc[:, :-self.label_count]
    
    def get_y(self):
        return self.df.iloc[:, -self.label_count:]
    
    def one_hot_encode_target(self):
        if self.df is None:
            print("Data not loaded. Please load the data first.")
            return None

        # One-hot encode the target variable
        encoder = OneHotEncoder(sparse=False)
        y_onehot = encoder.fit_transform(self.df.iloc[:, -1].values.reshape(-1, 1))

        # Update the label_count
        self.label_count = y_onehot.shape[1]

        # Convert one-hot encoded array back to DataFrame for easier manipulation
        y_onehot_df = pd.DataFrame(y_onehot, columns=encoder.get_feature_names_out(input_features=['target']))

        # Concatenate features and one-hot encoded target
        preprocessed_data = pd.concat([self.get_X(), y_onehot_df], axis=1)
        
        return preprocessed_data
