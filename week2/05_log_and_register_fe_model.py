# Databricks notebook source
import mlflow

# from life_expectancy.config import ProjectConfig
from databricks import feature_engineering
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

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
