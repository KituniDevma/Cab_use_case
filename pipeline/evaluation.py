from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)
import xgboost as xgb


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the XGBoost model using various regression metrics.

    Args:
        model (xgb.Booster): Trained XGBoost model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True test labels.

    Returns:
        dict: A dictionary containing all evaluation metrics.
    """
    # Convert test data to DMatrix
    test_dmatrix = xgb.DMatrix(data=X_test)

    # Predict the values
    y_pred = model.predict(test_dmatrix)

    # Calculate evaluation metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)  # Root Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
    r2 = r2_score(y_test, y_pred)  # R² Score
    explained_var = explained_variance_score(y_test, y_pred)  # Explained Variance

    # Combine metrics into a dictionary
    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R² Score": r2,
        "Explained Variance": explained_var
    }

    return metrics
