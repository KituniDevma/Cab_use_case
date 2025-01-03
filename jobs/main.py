import sys
import os

# Append the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from pipeline.data_preprocessing import preprocess
from conf.config import init_spark, load_csv_data
from pipeline.training import train_model_pipeline
from pipeline.evaluation import evaluate_model
from conf.parameters import parameters
import logging

# Initialize logger
logger = logging.getLogger("MainPipelineLogger")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def main():
    """
    Main function to run the data preprocessing, model training, and evaluation pipeline.
    """
    try:
        logger.info("Pipeline execution started.")

        # Initialize Spark session
        logger.info("Initializing Spark session.")
        spark = init_spark()
        
        # Load data
        logger.info("Loading data from CSV file.")
        df_spark = load_csv_data(spark, parameters["paths"]["data_path"])
        logger.info(f"Data loaded successfully with {df_spark.count()} rows and {len(df_spark.columns)} columns.")

        # Preprocess the data
        logger.info("Preprocessing the data.")
        processed_data = preprocess(df_spark)
        logger.info("Data preprocessing completed.")

        # Convert Spark DataFrame to Pandas
        logger.info("Converting Spark DataFrame to Pandas DataFrame.")
        df = processed_data.toPandas()
        logger.info(f"Converted DataFrame shape: {df.shape}")

        # Target column
        target_column = parameters["train_test_split"]["target_column"]
        logger.info(f"Target column for training: {target_column}")

        # Train the model
        logger.info("Starting model training pipeline.")
        model, splits = train_model_pipeline(df, target_column)
        logger.info("Model training completed successfully.")

        # Evaluate the model
        logger.info("Evaluating the model.")
        X_train, X_test, y_train, y_test = splits
        metrics = evaluate_model(model, X_test, y_test)
        logger.info("Model evaluation completed.")

        # Print evaluation metrics
        logger.info("Model Evaluation Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        logger.info("Pipeline execution completed successfully.")
    
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()
