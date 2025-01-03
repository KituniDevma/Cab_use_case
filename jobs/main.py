import sys
import os

# Append the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)


from conf.config import init_spark, load_csv_data
from pipeline.training import train_model_pipeline
from pipeline.evaluation  import evaluate_model
from conf.parameters import parameters
import pandas as pd

def main():
    # Initialize Spark and load data
    spark = init_spark()
    df_spark = load_csv_data(spark, parameters["paths"]["data_path"])

    # Convert Spark DataFrame to Pandas
    df = df_spark.toPandas()

    # Target column
    target_column = parameters["train_test_split"]["target_column"]

    # Train the model
    df_train = load_csv_data(spark, parameters["paths"]["output_path"])
    df_train = df_train.toPandas()
    print(df_train.columns)
    model, splits = train_model_pipeline(df_train, target_column)

    # Evaluate the model
    X_train, X_test, y_train, y_test = splits
    metrics = evaluate_model(model, X_test, y_test)

    # Print all evaluation metrics
    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
