from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import xgboost as xgb
import logging

# Initialize logger
logger = logging.getLogger("ModelEvaluationLogger")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def evaluate_xgboost_model(model, X_test, y_test):
    """
    Evaluate the XGBoost model using various regression metrics.

    Args:
        model (xgb.Booster): Trained XGBoost model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True test labels.

    Returns:
        dict: A dictionary containing all evaluation metrics.
    """
    try:
        logger.info("Starting model evaluation.")

        # Convert test data to DMatrix
        logger.info("Converting test data to DMatrix format for XGBoost.")
        test_dmatrix = xgb.DMatrix(data=X_test)

        # Predict the values
        logger.info("Generating predictions using the trained model.")
        y_pred = model.predict(test_dmatrix)

        # Calculate evaluation metrics
        logger.info("Calculating evaluation metrics.")
        rmse = root_mean_squared_error(y_test, y_pred)  # Root Mean Squared Error
        mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
        r2 = r2_score(y_test, y_pred)  # R² Score
        explained_var = explained_variance_score(y_test, y_pred)  # Explained Variance

        # Log calculated metrics
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"R² Score: {r2:.4f}")
        logger.info(f"Explained Variance: {explained_var:.4f}")

        # Combine metrics into a dictionary
        metrics = {
            "RMSE": rmse,
            "MAE": mae,
            "R² Score": r2,
            "Explained Variance": explained_var
        }

        logger.info("Model evaluation completed successfully.")
        return metrics

    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise
