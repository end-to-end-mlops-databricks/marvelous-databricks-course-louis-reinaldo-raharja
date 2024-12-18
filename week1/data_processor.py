import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataProcessor:
    def __init__(self, filepath, config):
        self.df = self.load_data(filepath)
        self.config = config
        self.X = None
        self.y = None
        self.preprocessor = None

    def load_data(self, filepath):
        return pd.read_csv(filepath)

    def preprocess_data(self):
        # Remove rows with missing target
        target = self.config["target"]
        self.df = self.df.dropna(subset=[target])

        # Separate features and target
        self.X = self.df[self.config["num_features"] + self.config["cat_features"]]
        self.y = self.df[target]

        # Create preprocessing steps for numeric and categorical data [[1]]
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),  # Changed to mean for life expectancy data
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),  # Changed to most_frequent
                ("onehot", OneHotEncoder(drop="first", sparse_output=False)),  # Modified for better handling
            ]
        )

        # Combine preprocessing steps [[4]]
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.config["num_features"]),
                ("cat", categorical_transformer, self.config["cat_features"]),
            ]
        )

    def split_data(self, test_size=0.2, random_state=42):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
