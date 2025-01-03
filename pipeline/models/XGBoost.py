import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from conf.parameters import parameters


def train_xgboost_model(df: pd.DataFrame, target_column: str):
    """
    Train an XGBoost model on the dataset.

    Args:
        df (pd.DataFrame): Preprocessed dataset.
        target_column (str): The column to predict.

    Returns:
        xgboost.Booster: The trained XGBoost model.
    """
    try:
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=parameters["train_test_split"]["test_size"], random_state=parameters["train_test_split"]["random_state"]
        )

        # Convert data to DMatrix for XGBoost
        train_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
        test_dmatrix = xgb.DMatrix(data=X_test, label=y_test)

        # Train the model
        logger.info("Training the XGBoost model...")
        model = xgb.train(params=parameters["model_params"], dtrain=train_dmatrix)
        logger.info("Model training completed.")

        # Evaluate the model
        y_pred = model.predict(test_dmatrix)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        logger.info(f"Model evaluation completed. RMSE: {rmse}")

        return model

    except Exception as e:
        logger.error(f"Error in training the XGBoost model: {e}")
        raise
