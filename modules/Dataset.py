import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

class Dataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = self.load_data()
        self.label_count = 1  # Initialize with 1 assuming only one target column before encoding

        self.preprocess_data()

    def preprocess_data(self):
        self.handle_string_features()
        self.df, self.label_count = self.one_hot_encode_target()
        self.df = self.normalize_data()
        self.df = self.df.sample(frac=1).reset_index(drop=True)

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
        return self.df.iloc[:, :-self.label_count]
    
    def get_y(self):
        return self.df.iloc[:, -self.label_count:]
    
    def one_hot_encode_target(self):
        encoder = OneHotEncoder(sparse=False)
        y_onehot = encoder.fit_transform(self.df.iloc[:, -1].values.reshape(-1, 1))
        label_count = y_onehot.shape[1]
        y_onehot_df = pd.DataFrame(y_onehot, columns=encoder.get_feature_names_out(input_features=['target']))
        
        # Concatenate the original features with the one-hot encoded target labels
        preprocessed_data = pd.concat([self.df.iloc[:, :-1], y_onehot_df], axis=1)
        
        return preprocessed_data, label_count

    def normalize_data(self):
        # Separate the features and target labels
        X = self.df.iloc[:, :-self.label_count]
        y = self.df.iloc[:, -self.label_count:]

        # Normalize the features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Concatenate the normalized features with the target labels
        normalized_data = pd.concat([X_scaled_df, y], axis=1)
        
        return normalized_data
