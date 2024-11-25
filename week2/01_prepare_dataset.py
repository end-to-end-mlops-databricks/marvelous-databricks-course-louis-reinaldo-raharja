from databricks.connect import DatabricksSession

from life_expectancy.config import ProjectConfig
from life_expectancy.data_processor import DataProcessor

spark = DatabricksSession.builder.profile("dbc-643c4c2b-d6c9").getOrCreate()

# COMMAND ----------

config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# COMMAND ----------
# Load the house prices dataset
df = spark.sql("SELECT * FROM mlops_students.louisreinaldo.life_expectancy").toPandas()
print("query completed!")

# COMMAND ----------
data_processor = DataProcessor(df=df, config=config)
data_processor.preprocess_data()
train_set, test_set = data_processor.split_data()
data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)
print("data processor complete!")
