import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_data(dataset_path):
    data = pd.read_csv(dataset_path, header=None)  # Assuming the dataset doesn't have a header
    return data

def preprocess_data(data):
    # Separate features and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # One-hot encode the target variable
    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(y.values.reshape(-1, 1))
    
    # Convert one-hot encoded array back to DataFrame for easier manipulation
    y_onehot_df = pd.DataFrame(y_onehot, columns=encoder.get_feature_names_out(input_features=['target']))
    
    # Concatenate features and one-hot encoded target
    preprocessed_data = pd.concat([X, y_onehot_df], axis=1)
    
    return preprocessed_data

