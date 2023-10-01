import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np

class Dataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = self.load_data()
        self.label_count = 1  # Initialize with 1 assuming only one target column before encoding
        self.feature_count = self.df.shape[1]


        self._X_numpy = self.df.iloc[:, :-self.label_count].to_numpy()
        self._y_numpy = self.df.iloc[:, -self.label_count:].to_numpy()

        #print(self.df.head())
        self.preprocess_data()
        #print(self.df.head())

        self.split_data()


    def preprocess_data(self):
        self.handle_string_features()
        self.df, self.label_count = self.one_hot_encode_target()
        
        # Shuffle the dataset using NumPy
        np.random.shuffle(self.df)
        
        # Update the instance variables
        self._X_numpy = self.df[:, :-self.label_count]
        self._y_numpy = self.df[:, -self.label_count:]


    def handle_string_features(self):
        label_encoders = {}  # To keep track of label encoders for each column (if needed later)
        
        for col in self.df.columns:
            if self.df[col].dtype == 'object':  # Check if the column is of object type (usually for strings)
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                label_encoders[col] = le  # Store the label encoder
                
        return label_encoders

    def get_df(self):
        return self.df
    
    def get_label_count(self):
        return self.label_count

    def load_data(self):
        data = pd.read_csv(self.dataset_path, header=None)
        return data
    
    def get_X(self):
        return self._X_numpy
    
    def get_y(self):
        return self._y_numpy
    
    def one_hot_encode_target(self):
        encoder = OneHotEncoder(sparse=False)
        y_onehot = encoder.fit_transform(self.df.iloc[:, -1].values.reshape(-1, 1))
        label_count = y_onehot.shape[1]
        
        # Convert the original DataFrame to NumPy array and remove the last column (target)
        data_numpy = self.df.iloc[:, :-1].to_numpy()
        
        # Concatenate the original features with the one-hot encoded target labels
        preprocessed_data = np.hstack([data_numpy, y_onehot])
        
        # Update the instance variables
        self._X_numpy = preprocessed_data[:, :-label_count]
        self._y_numpy = preprocessed_data[:, -label_count:]
        
        return preprocessed_data, label_count

    def normalize_data(self):
        # Normalize the features using MinMaxScaler
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(self._X_numpy)

        # Concatenate the normalized features with the target labels
        normalized_data = np.hstack([X_scaled, self._y_numpy])

        # Update the instance variables
        self._X_numpy = normalized_data[:, :-self.label_count]
        self._y_numpy = normalized_data[:, -self.label_count:]

        return normalized_data


    def split_data(self, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self._X_numpy, self._y_numpy, test_size=test_size, random_state=42
        )

    def set_new_data(self, X, y):
        self._X_numpy = X
        self._X_numpy = y