import logging
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Initialize logger
logger = logging.getLogger("XGBoostModelLogger")

def train_xgboost_model(df, target_column, model_params, test_size, random_state):
    """
    Prepare the dataset and train an XGBoost model.

    Args:
        df (pd.DataFrame): Preprocessed dataset.
        target_column (str): The column to predict.
        model_params (dict): Parameters for the XGBoost model.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.

    Returns:
        model: Trained XGBoost model.
        tuple: Train-test data splits (X_train, X_test, y_train, y_test).
    """
    try:
        logger.info("Starting XGBoost model training process.")
        
        # Separate features and target
        logger.info(f"Separating features and target column: {target_column}.")
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        logger.debug(f"Feature columns: {X.columns.tolist()}. Target column: {target_column}.")

        # Train-test split
        logger.info(f"Splitting data into train and test sets with test_size={test_size}, random_state={random_state}.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        logger.info("Data split completed successfully.")
        logger.debug(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}.")

        # Convert data to DMatrix for XGBoost
        logger.info("Converting training data to DMatrix format for XGBoost.")
        train_dmatrix = xgb.DMatrix(data=X_train, label=y_train)

        # Train the model
        logger.info("Training XGBoost model with provided parameters.")
        model = xgb.train(params=model_params, dtrain=train_dmatrix)
        logger.info("Model training completed successfully.")

        return model, (X_train, X_test, y_train, y_test)

    except Exception as e:
        logger.error(f"Error during XGBoost model training: {e}")
        raise

