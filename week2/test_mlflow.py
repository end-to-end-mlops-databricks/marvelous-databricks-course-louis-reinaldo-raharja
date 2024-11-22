# Databricks notebook source
import json
import subprocess
import mlflow

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_name="/Shared/life-expectancy-basic")
# mlflow.set_experiment_tags({"repository_name": "life-expectancy"})