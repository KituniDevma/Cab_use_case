import pandas as pd
from conf.parameters import parameters
from pipeline.models.XGBoost import train_xgboost_model

def train_model_pipeline(df, target_column):
    """
    Train the XGBoost model pipeline.

    Args:
        df (pd.DataFrame): Preprocessed dataset.
        target_column (str): The column to predict.

    Returns:
        model: Trained model.
        tuple: Data splits (X_train, X_test, y_train, y_test).
    """
    model, splits = train_xgboost_model(
        df,
        target_column,
        model_params=parameters["xgboost"],
        test_size=parameters["train_test_split"]["test_size"],
        random_state=parameters["train_test_split"]["random_state"],
    )
    return model, splits
