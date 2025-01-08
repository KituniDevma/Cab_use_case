import logging
from pyspark.ml.feature import VectorAssembler
from pipeline.models.LinearRegression import create_linear_regression_model

# Initialize logger
logger = logging.getLogger("TrainingLogger")

def train_regression_model_pipeline(df, target_column):
    """
    Trains the Linear Regression model using PySpark MLlib.

    Args:
        df (DataFrame): Spark DataFrame containing features and target.
        target_column (str): Column name for the target variable.

    Returns:
        LinearRegressionModel: Trained Linear Regression model.
        DataFrame: Predictions DataFrame.
    """
    try:
        logger.info("Starting training process.")

        # Assemble features
        logger.info("Assembling features into a single vector column.")
        feature_columns = [col for col in df.columns if col != target_column]
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        df = assembler.transform(df).select("features", target_column)

        # Initialize Linear Regression model
        model = create_linear_regression_model()
        logger.info("Training Linear Regression model.")
        
        # Fit the model
        linear_model = model.fit(df.withColumnRenamed(target_column, "label"))
        logger.info("Model training completed successfully.")

        # Make predictions
        predictions = linear_model.transform(df)

        return predictions

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise
