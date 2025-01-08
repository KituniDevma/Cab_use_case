import logging
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize logger
logger = logging.getLogger("LinearRegressionEvaluationLogger")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def evaluate_regression_model(predictions, target_column):
    """
    Evaluate the Linear Regression model using various regression metrics.

    Args:
        model: Trained Linear Regression model.
        predictions (DataFrame): DataFrame containing predictions and true labels.

    Returns:
        dict: A dictionary containing all evaluation metrics.
    """
    try:
        logger.info("Starting Linear Regression model evaluation.")
        print(target_column)

        # Evaluate using RegressionEvaluator
        evaluator = RegressionEvaluator(predictionCol="prediction", labelCol=target_column)

        rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
        mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
        r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

        # Log calculated metrics
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"R² Score: {r2:.4f}")

        # Combine metrics into a dictionary
        metrics = {
            "RMSE": rmse,
            "MAE": mae,
            "R² Score": r2
        }

        logger.info("Model evaluation completed successfully.")
        return metrics

    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise
