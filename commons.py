import numpy as np
import pandas as pd
import pickle
import feature_eng
from sklearn.base import BaseEstimator, TransformerMixin

def load_model(file_path):
    with open(file_path, mode='rb') as my_file:
        model = pickle.load(my_file)
    return model

def extract_diabetes_features(test_data):
    selector = load_model("models/selector.pkl")
    scaler = load_model("models/corr_scaler.pkl")
    normalizer = load_model(file_path='models/normalizer.pkl')

    selected_names = test_data.columns[selector.get_support()]
    features = pd.DataFrame(data=selector.transform(test_data), columns=selected_names)
    features = scaler.transform(features)
    return normalizer.transform(features)


def extract_bp_features(test_data):
    # Load the feature selector and scaler
    selector = load_model("bp_models/bp_selector.pkl")
    scaler = load_model("bp_models/bp_scaler.pkl")

    # Select features using the selector
    selected_names = test_data.columns[selector.get_support()]
    features = pd.DataFrame(data=selector.transform(test_data), columns=selected_names)

    # Scale the features
    scaled_features = scaler.transform(features)
    return scaled_features


def perform_diabetes_test(test_data):
    # Feature selection for diabetes test
    features = extract_diabetes_features(test_data)

    # Load the models
    encoding = {0: "Low", 1: "Medium", 2: "High"}

    stage_model = load_model('models/stage.pkl')
    bgl_model = load_model('models/AdaBoost.pkl')

    stage = stage_model.predict(features)[0]
    bgl = bgl_model.predict(features)[0]

    return encoding[stage], np.round(bgl, 2)
    
def perform_bp_test(test_data):
    # Perform feature selection for BP test
    test_data = test_data.drop(columns=['maxBP', 'minBP'], errors='ignore', axis=1)
    features = extract_bp_features(test_data)

    # Load the models for BP prediction
    max_model = load_model('bp_models/max.pkl')
    min_model = load_model('bp_models/min.pkl')
    max_value = max_model.predict(features)[0]
    min_value = min_model.predict(features)[0]
    return np.round(max_value), np.round(min_value)

def remove_irrelevant_data(df):
    # Read the CSV file into a DataFrame, skipping the first 3 rows and setting the 4th row as header
    df.columns = ['H', 'MQ138', 'MQ2', 'SSID', 'T', 'TGS2600', 'TGS2602', 'TGS2603', 'TGS2610', 'TGS2611', 'TGS2620', 'TGS822', 'Device', 'Time']
    df = df.drop(['SSID', 'Device', 'H', 'T', 'Time'], axis=1)
    return df.reset_index(drop=True)

def generate_data(sensors_data, body_vitals):
    cleaned_df = remove_irrelevant_data(sensors_data)
    features_df = feature_eng.generate_features(df=cleaned_df)
    final_df = pd.concat([body_vitals, features_df], axis=1)
    return final_df

class RemoveHighlyCorrelatedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.to_drop_ = None

    def fit(self, X, y=None):
        # Compute the Spearman correlation matrix
        corr_matrix = X.corr(method='spearman').abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Identify correlated feature pairs
        correlated_pairs = [(col, row) for col in upper.columns for row in upper.index if upper.at[row, col] >= self.threshold]
        
        # Determine which feature to drop based on variance
        drop_set = set()
        for col, row in correlated_pairs:
            if X[col].var() >= X[row].var():
                drop_set.add(row)  # Drop the feature with lower variance
            else:
                drop_set.add(col)
        
        self.to_drop_ = list(drop_set)
        return self

    def transform(self, X, y=None):
        return X.drop(columns=self.to_drop_, errors='ignore')

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

