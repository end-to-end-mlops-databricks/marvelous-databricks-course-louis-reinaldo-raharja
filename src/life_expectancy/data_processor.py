import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from datetime import datetime
from life_expectancy.config import ProjectConfig


class DataProcessor:
    def __init__(self, df, config):
        self.df = df
        self.config = config
        self.X = None
        self.y = None
        self.preprocessor = None

    def load_data(self, filepath):
        return pd.read_csv(filepath)

    def preprocess_data(self):
        # Remove rows with missing target
        target = self.config.target
        self.df = self.df.dropna(subset=[target])

        # Separate features and target
        all_features = self.config.num_features + self.config.cat_features
        self.X = self.df[all_features]
        self.y = self.df[target]

        # Create preprocessing steps for numeric and categorical data
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop="first", sparse_output=False)),
            ]
        )

        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.config.num_features),
                ("cat", categorical_transformer, self.config.cat_features),
            ]
        )
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split the DataFrame (self.df) into training and test sets."""
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession):
        """Save the train and test sets into Databricks tables."""

        catalog_name = self.config.catalog_name
        schema_name = self.config.schema_name
        table_prefix = 'who_life_expectancy'

        spark.sql(f"USE CATALOG {catalog_name}")
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
        spark.sql(f"DROP TABLE IF EXISTS {catalog_name}.{schema_name}.{table_prefix}_train")
        spark.sql(f"DROP TABLE IF EXISTS {catalog_name}.{schema_name}.{table_prefix}_test")

        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC"))   
        
        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC"))

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{catalog_name}.{schema_name}.{table_prefix}_train")
        
        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{catalog_name}.{schema_name}.{table_prefix}_test")

        spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.{table_prefix}_train SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
        
        spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.{table_prefix}_test SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")