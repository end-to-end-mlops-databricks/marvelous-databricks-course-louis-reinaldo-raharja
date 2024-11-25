import json
import subprocess

import mlflow
import pandas as pd
from databricks.connect import DatabricksSession
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env

from life_expectancy.data_processor import ProjectConfig


def adjust_predictions(predictions, scale_factor=1.3):
    return predictions * scale_factor


def get_git_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()[:7]


git_sha = get_git_sha()

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")
client = MlflowClient()

config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
parameters = config.parameters
catalog_name = "mlops_students"
spark_table_name = "hive_metastore"
schema_name = config.schema_name
table_prefix = "life_expectancy"


spark = DatabricksSession.builder.profile("dbc-643c4c2b-d6c9").getOrCreate()


run_id = mlflow.search_runs(
    experiment_names=["/Shared/life-expectancy"],
    filter_string="tags.branch='week2'",
).run_id[0]

model = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-pipeline-model")


class LifeExpectancyModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            predictions = self.model.predict(model_input)
            predictions = {"Prediction": adjust_predictions(predictions[0])}
            return predictions
        else:
            raise ValueError("Input must be a pandas DataFrame.")


train_set = spark.table(f"{spark_table_name}.{schema_name}.{table_prefix}_train")
test_set = spark.table(f"{spark_table_name}.{schema_name}.{table_prefix}_test")

X_train = train_set[num_features + cat_features].toPandas()
y_train = train_set[[target]].toPandas()

X_test = test_set[num_features + cat_features].toPandas()
y_test = test_set[[target]].toPandas()

wrapped_model = LifeExpectancyModelWrapper(model)  # we pass the loaded model to the wrapper
example_input = X_test.iloc[0:1]  # Select the first row for prediction as example
example_prediction = wrapped_model.predict(context=None, model_input=example_input)
print("Example Prediction:", example_prediction)


mlflow.set_experiment(experiment_name="/Shared/life-expectancy-pyfunc")


with mlflow.start_run(tags={"branch": "week2", "git_sha": f"{git_sha}"}) as run:
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train, model_output={"Prediction": example_prediction})
    dataset = mlflow.data.from_spark(
        train_set, table_name=f"{spark_table_name}.{schema_name}.{table_prefix}_train", version="0"
    )
    mlflow.log_input(dataset, context="training")
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=[
            "../mlops_with_databricks-0.0.1-py3-none-any.whl",
        ],
        additional_conda_channels=None,
    )
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="pyfunc-life-expectancy-model",
        code_paths=["../mlops_with_databricks-0.0.1-py3-none-any.whl"],
        signature=signature,
    )

# COMMAND ----------
loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/pyfunc-life-expectancy-model")
loaded_model.unwrap_python_model()

# COMMAND ----------
model_name = f"{catalog_name}.{schema_name}.life_expectancy_model_pyfunc"

model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/pyfunc-life-expectancy-model", name=model_name, tags={"git_sha": f"{git_sha}"}
)
# COMMAND ----------

with open("model_version.json", "w") as json_file:
    json.dump(model_version.__dict__, json_file, indent=4)

# COMMAND ----------
model_version_alias = "the_best_model"
client.set_registered_model_alias(model_name, model_version_alias, "1")

model_uri = f"models:/{model_name}@{model_version_alias}"
model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------
client.get_model_version_by_alias(model_name, model_version_alias)
