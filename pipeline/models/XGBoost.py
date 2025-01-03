import xgboost as xgb
from sklearn.model_selection import train_test_split

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

    # Separate features and target
    X = df.drop(target_column)
    y = df[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Convert data to DMatrix for XGBoost
    train_dmatrix = xgb.DMatrix(data=X_train, label=y_train)

    # Train the model
    model = xgb.train(params=model_params, dtrain=train_dmatrix)

    return model, (X_train, X_test, y_train, y_test)
