from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
import xgboost as xgb

def train_model_pipeline(spark, data, parameters):
    """
    Train and evaluate the XGBoost model using PySpark.

    Args:
        spark (SparkSession): The externally initialized Spark session.
        data: Preprocessed Spark DataFrame.
        parameters: Configuration parameters for training and evaluation.
    """
    # Vectorize the features for XGBoost
    feature_cols = [col for col in data.columns if col not in ("Total_Amount", "Tip", "fare")]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data = assembler.transform(data)

    # Convert Spark DataFrame to Pandas DataFrame for XGBoost
    pandas_df = data.select("features", "fare").toPandas()

    X = pandas_df["features"].tolist()
    y = pandas_df["fare"]

    # Train-test split
    test_size = parameters["train_test"]["test_size"]
    random_state = parameters["train_test"]["random_state"]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train the XGBoost model
    model_params = parameters["xgboost"]
    model = xgb.XGBRegressor(**model_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

    # Save the trained model
    model_output_path = f"{parameters['paths']['output_path']}/trained_model.json"
    model.save_model(model_output_path)
