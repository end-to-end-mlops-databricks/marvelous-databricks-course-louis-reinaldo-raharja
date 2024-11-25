# Databricks notebook source
import json
import subprocess

import mlflow

mlflow.set_tracking_uri("databricks")

mlflow.set_experiment(experiment_name="/Shared/life-expectancy-basic")
mlflow.set_experiment_tags({"repository_name": "life-expectancy"})


def get_git_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()[:7]


git_sha = get_git_sha()
print(f"git_sha: {git_sha}")

# COMMAND ----------
experiments = mlflow.search_experiments(filter_string="tags.repository_name='life-expectancy'")
print(experiments)

# COMMAND ----------
with open("mlflow_experiment.json", "w") as json_file:
    json.dump(experiments[0].__dict__, json_file, indent=4)
# COMMAND ----------
with mlflow.start_run(
    run_name="demo-run",
    tags={"git_sha": git_sha, "branch": "week2"},
    description="demo run",
) as run:
    mlflow.log_params({"type": "demo"})
    mlflow.log_metrics({"metric1": 1.0, "metric2": 2.0})
# COMMAND ----------
run_id = mlflow.search_runs(
    experiment_names=["/Shared/life-expectancy-basic"],
    filter_string=f"tags.git_sha='{git_sha}'",
).run_id[0]
run_info = mlflow.get_run(run_id=f"{run_id}").to_dictionary()
print(run_info)

# COMMAND ----------
with open("run_info.json", "w") as json_file:
    json.dump(run_info, json_file, indent=4)

# COMMAND ----------
print(run_info["data"]["metrics"])

# COMMAND ----------
print(run_info["data"]["params"])
