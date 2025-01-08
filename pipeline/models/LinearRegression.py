from pyspark.ml.regression import LinearRegression

def create_linear_regression_model():
    """
    Creates and returns a Linear Regression model using PySpark MLlib.
    
    Returns:
        LinearRegression: Linear Regression model instance.
    """
    return LinearRegression(featuresCol='features', labelCol='label')