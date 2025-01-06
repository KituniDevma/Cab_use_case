import pandas as pd
import logging
from conf.parameters import parameters
from pipeline.models.XGBoost import train_xgboost_model

# Initialize logger
logger = logging.getLogger("ModelPipelineLogger")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def train_xgboost_model_pipeline(df, target_column):
    """
    Train the XGBoost model pipeline.

    Args:
        df (pd.DataFrame): Preprocessed dataset.
        target_column (str): The column to predict.

    Returns:
        model: Trained model.
        tuple: Data splits (X_train, X_test, y_train, y_test).
    """
    try:
        logger.info("Starting the model training pipeline.")
        
        # Logging input details
        logger.info(f"Target column: {target_column}")
        logger.info(f"Dataframe shape: {df.shape}")
        logger.info("Training parameters:")
        logger.info(f"  XGBoost parameters: {parameters['xgboost']}")
        logger.info(f"  Test size: {parameters['train_test_split']['test_size']}")
        logger.info(f"  Random state: {parameters['train_test_split']['random_state']}")
        
        # Train the model
        logger.info("Calling the train_xgboost_model function.")
        model, splits = train_xgboost_model(
            df,
            target_column,
            model_params=parameters["xgboost"],
            test_size=parameters["train_test_split"]["test_size"],
            random_state=parameters["train_test_split"]["random_state"],
        )
        
        # Log success and output details
        logger.info("Model training completed successfully.")
        X_train, X_test, y_train, y_test = splits
        logger.info(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
        logger.info(f"Test data shape: X_test={X_test.shape}, y_test={y_test.shape}")
        
        return model, splits
    
    except Exception as e:
        logger.error(f"Error during model training pipeline: {e}")
        raise
