# Databricks notebook source
import mlflow

# from life_expectancy.config import ProjectConfig
import pandas as pd
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# COMMAND ----------

# Initialize clients
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# Setup MLflow
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# From config file
num_features = [
    "Adult_Mortality",
    "infant_deaths",
    "Alcohol",
    "percentage_expenditure",
    "Hepatitis_B",
    "Measles",
    "BMI",
    "under-five_deaths",
    "Polio",
    "Total_expenditure",
    "Diphtheria",
    "HIV_AIDS",
    "GDP",
    "Population",
    "thinness_1-19_years",
    "thinness_5-9_years",
    "Income_composition_of_resources",
    "Schooling",
]

cat_features = ["Status"]

target = "Life_expectancy"

parameters = {"learning_rate": 0.01, "n_estimators": 100, "max_depth": 10}

# Manual settings
catalog_name = "mlops_students"
spark_table_name = "hive_metastore"
schema_name = "louisreinaldo"
table_prefix = "life_expectancy"

# Create feature table
feature_table_name = f"{catalog_name}.{schema_name}.health_features"
function_name = f"{catalog_name}.{schema_name}.calculate_health_index"

# Create feature table
spark.sql(f"""
CREATE OR REPLACE TABLE {feature_table_name}
(Country STRING NOT NULL,
 BMI DOUBLE,
 HIV_AIDS DOUBLE,
 Total_expenditure DOUBLE);
""")

spark.sql(f"ALTER TABLE {feature_table_name} ADD CONSTRAINT health_pk PRIMARY KEY(Country);")
spark.sql(f"ALTER TABLE {feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

# Insert data into feature table
print("insert into feature table")
spark.sql(f"""
INSERT INTO {feature_table_name}
SELECT Country, BMI, HIV_AIDS, Total_expenditure
FROM {spark_table_name}.{schema_name}.{table_prefix}_train
UNION
SELECT Country, BMI, HIV_AIDS, Total_expenditure
FROM {spark_table_name}.{schema_name}.{table_prefix}_test
""")


# COMMAND ----------

# Create health index calculation function with NULL handling
spark.sql(f"""
CREATE OR REPLACE FUNCTION {function_name}(bmi DOUBLE, hiv DOUBLE, expenditure DOUBLE)
RETURNS DOUBLE
LANGUAGE PYTHON AS
$$
def calculate_index(bmi, hiv, expenditure):
    # Handle NULL values
    if bmi is None or hiv is None or expenditure is None:
        return None

    # Normalize and combine health indicators
    try:
        normalized_bmi = min(max(float(bmi), 18.5), 25) / 25
        hiv_factor = 1 - (float(hiv) / 100)
        exp_factor = float(expenditure) / 100
        return (normalized_bmi + hiv_factor + exp_factor) / 3
    except (TypeError, ValueError):
        return None

    # Normalize and combine health indicators
    try:
        normalized_bmi = min(max(float(bmi), 18.5), 25) / 25
        hiv_factor = 1 - (float(hiv) / 100)
        exp_factor = float(expenditure) / 100
        return (normalized_bmi + hiv_factor + exp_factor) / 3
    except (TypeError, ValueError):
        return None
return calculate_index(bmi, hiv, expenditure)
$$
""")

# COMMAND ----------

print("load dataset")
# Load datasets and drop the columns that will be looked up
train_set = spark.table(f"{spark_table_name}.{schema_name}.{table_prefix}_train").drop(
    "BMI", "HIV_AIDS", "Total_expenditure"
)
test_set = spark.table(f"{spark_table_name}.{schema_name}.{table_prefix}_test").toPandas()

# Make sure Country column is string type for lookup
train_set = train_set.withColumn("Country", train_set["Country"].cast("string"))

# COMMAND ----------

print("feature engineering setup")
# Feature engineering setup
# Modify the feature engineering setup to handle NULL values
training_set = fe.create_training_set(
    df=train_set,
    label=target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["BMI", "HIV_AIDS", "Total_expenditure"],
            lookup_key="Country",
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="health_index",
            input_bindings={"bmi": "BMI", "hiv": "HIV_AIDS", "expenditure": "Total_expenditure"},
        ),
    ],
    exclude_columns=["update_timestamp_utc"],
)

# Handle NULL values in the test set calculation
test_set["health_index"] = test_set.apply(
    lambda x: None
    if pd.isna(x["BMI"]) or pd.isna(x["HIV_AIDS"]) or pd.isna(x["Total_expenditure"])
    else (
        min(max(float(x["BMI"]), 18.5), 25) / 25
        + (1 - float(x["HIV_AIDS"]) / 100)
        + float(x["Total_expenditure"]) / 100
    )
    / 3,
    axis=1,
)

# COMMAND ----------

training_df = training_set.load_df().toPandas()
X_train = training_df[num_features + cat_features + ["health_index"]]
y_train = training_df[target]

X_test = test_set[num_features + cat_features + ["health_index"]]
y_test = test_set[target]

# COMMAND ----------

# Create and train pipeline
print("preprocess")

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)], remainder="passthrough"
)

print("creating pipeline")

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", LGBMRegressor(**parameters))])

# COMMAND ----------

# MLflow tracking
git_sha = "d4d2b46"
mlflow.set_experiment(experiment_name="/Shared/life-expectancy-fe")

# COMMAND ----------

print("start mlflow run")

with mlflow.start_run(tags={"branch": "week2", "git_sha": git_sha}) as run:
    run_id = run.info.run_id

    # Train and evaluate
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log metrics and parameters
    mlflow.log_param("model_type", "LightGBM with FE")
    mlflow.log_params(parameters)
    mlflow.log_metrics({"mse": mse, "mae": mae, "r2_score": r2})

    # Log model
    signature = infer_signature(model_input=X_train, model_output=y_pred)
    fe.log_model(
        model=pipeline,
        flavor=mlflow.sklearn,
        artifact_path="lightgbm-pipeline-model-fe",
        training_set=training_set,
        signature=signature,
    )

# COMMAND ----------

print("register model")
# Register model
mlflow.register_model(
    model_uri=f"runs:/{run_id}/lightgbm-pipeline-model-fe", name=f"{catalog_name}.{schema_name}.{table_prefix}_model_fe"
)

# COMMAND ----------
