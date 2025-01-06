from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import logging

# Initialize logger
logger = logging.getLogger("LinearRegressionEvaluationLogger")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def evaluate_regression_model(model, X_test, y_test):
    """
    Evaluate the Linear Regression model using various regression metrics.

    Args:
        model: Trained Linear Regression model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True test labels.

    Returns:
        dict: A dictionary containing all evaluation metrics.
    """
    try:
        logger.info("Starting Linear Regression model evaluation.")

        # Predict the values
        logger.info("Generating predictions using the trained model.")
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        logger.info("Calculating evaluation metrics.")
        rmse = mean_squared_error(y_test, y_pred)  # Root Mean Squared Error
        mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
        r2 = r2_score(y_test, y_pred)  # R² Score
        explained_var = explained_variance_score(y_test, y_pred)  # Explained Variance

        # Log calculated metrics
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"R² Score: {r2:.4f}")
        logger.info(f"Explained Variance: {explained_var:.4f}")

        # Combine metrics into a dictionary
        metrics2 = {
            "RMSE": rmse,
            "MAE": mae,
            "R² Score": r2,
            "Explained Variance": explained_var
        }

        logger.info("Model evaluation completed successfully.")
        return metrics2

    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise
