from databricks.connect import DatabricksSession
from databricks import feature_engineering
from databricks.sdk import WorkspaceClient
import mlflow
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from life_expectancy.config import ProjectConfig
import subprocess

# Initialize clients
spark = DatabricksSession.builder.profile("dbc-643c4c2b-d6c9").getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# Setup MLflow
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
num_features = config.num_features
cat_features = config.cat_features
target = config.target
parameters = config.parameters
catalog_name = 'mlops_students'
spark_table_name = 'hive_metastore'
schema_name = config.schema_name
table_prefix = 'life_expectancy'

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
print('insert into feature table')
spark.sql(f"""
INSERT INTO {feature_table_name}
SELECT Country, BMI, HIV_AIDS, Total_expenditure 
FROM {spark_table_name}.{schema_name}.{table_prefix}_train
UNION
SELECT Country, BMI, HIV_AIDS, Total_expenditure 
FROM {spark_table_name}.{schema_name}.{table_prefix}_test
""")

# Create health index calculation function
spark.sql(f"""
CREATE OR REPLACE FUNCTION {function_name}(bmi DOUBLE, hiv DOUBLE, expenditure DOUBLE)
RETURNS DOUBLE
LANGUAGE PYTHON AS
$$
def calculate_index(bmi, hiv, expenditure):
    # Normalize and combine health indicators
    normalized_bmi = min(max(bmi, 18.5), 25) / 25
    hiv_factor = 1 - (hiv / 100)
    exp_factor = expenditure / 100
    return (normalized_bmi + hiv_factor + exp_factor) / 3

return calculate_index(bmi, hiv, expenditure)
$$
""")

# Load datasets
print('load dataset')
train_set = spark.table(f"{spark_table_name}.{schema_name}.{table_prefix}_train")
test_set = spark.table(f"{spark_table_name}.{schema_name}.{table_prefix}_test").toPandas()

# Feature engineering setup
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
            input_bindings={
                "bmi": "BMI",
                "hiv": "HIV_AIDS",
                "expenditure": "Total_expenditure"
            },
        ),
    ],
    exclude_columns=["update_timestamp_utc"]
)

# Load and prepare data
training_df = training_set.load_df().toPandas()
X_train = training_df[num_features + cat_features + ["health_index"]]
y_train = training_df[target]

# Calculate health index for test set
test_set["health_index"] = test_set.apply(
    lambda x: (min(max(x["BMI"], 18.5), 25) / 25 + 
              (1 - x["HIV_AIDS"] / 100) + 
              x["Total_expenditure"] / 100) / 3, 
    axis=1
)
X_test = test_set[num_features + cat_features + ["health_index"]]
y_test = test_set[target]

# Create and train pipeline
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)],
    remainder="passthrough"
)
pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", LGBMRegressor(**parameters))]
)

# MLflow tracking
git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()[:7]
mlflow.set_experiment(experiment_name="/Shared/life-expectancy-fe")

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
    mlflow.log_metrics({
        "mse": mse,
        "mae": mae,
        "r2_score": r2
    })
    
    # Log model
    signature = infer_signature(model_input=X_train, model_output=y_pred)
    fe.log_model(
        model=pipeline,
        flavor=mlflow.sklearn,
        artifact_path="lightgbm-pipeline-model-fe",
        training_set=training_set,
        signature=signature,
    )

# Register model
mlflow.register_model(
    model_uri=f'runs:/{run_id}/lightgbm-pipeline-model-fe',
    name=f"{catalog_name}.{schema_name}.{table_prefix}_model_fe"
)