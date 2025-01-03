from pyspark.sql import functions as F

def preprocess(data):
    """
    Preprocess the data: create new feature 'fare', and encode categorical variables.

    Args:
        data: Spark DataFrame to preprocess.

    Returns:
        DataFrame: Preprocessed Spark DataFrame with all original columns and new features.
    """
    # Create the 'fare' column by subtracting 'tip' from 'total_amount'
    data = data.withColumn("Fare", F.col("Total_Amount") - F.col("Tip"))

    # Encode the 'gender' column: 'F' -> 0, 'M' -> 1
    data = data.withColumn("Gender_Encoded", F.when(F.col("Gender") == "F", 0).otherwise(1))

    # Encode the 'PickUp_Time' column: 'day' -> 0, 'night' -> 1
    data = data.withColumn("PickUp_Time_Encoded", F.when(F.col("PickUp_Time") == "day", 0).otherwise(1))

    # Drop the original columns used for new feature creation
    data = data.drop("Tip", "Gender", "PickUp_Time", "Total_Amount")

    return data
