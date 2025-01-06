# training.py
import logging
from sklearn.model_selection import train_test_split
from pipeline.models.LinearRegression import create_linear_regression_model

# Initialize logger
logger = logging.getLogger("TrainingLogger")

def train_regression_model_pipeline(df, target_column, test_size=0.2, random_state=42):
    """
    Trains the Linear Regression model.

    Args:
        df (pd.DataFrame): Dataset containing features and target.
        target_column (str): Column name for the target variable.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        model: Trained Linear Regression model.
        tuple: Train-test splits (X_train, X_test, y_train, y_test).
    """
    try:
        logger.info("Starting training process.")

        # Separate features and target
        logger.info(f"Separating features and target: {target_column}")
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        logger.debug(f"Feature columns: {X.columns.tolist()}")

        # Train-test split
        logger.info("Splitting data into train and test sets.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        logger.debug(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

        # Create and train the model
        model = create_linear_regression_model()
        logger.info("Training Linear Regression model.")
        model.fit(X_train, y_train)
        logger.info("Model training completed successfully.")

        return model, (X_train, X_test, y_train, y_test)

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise
