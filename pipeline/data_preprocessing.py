from pyspark.sql import functions as F

def preprocess(data):
    """
    Preprocess the data: create new feature 'fare', and encode categorical variables.

    Args:
        data: Spark DataFrame to preprocess.

    Returns:
        DataFrame: Preprocessed Spark DataFrame.
    """
    # Create the 'fare' column by subtracting 'tip' from 'total_amount'
    data = data.withColumn("fare", F.col("Total_Amount") - F.col("Tip"))

    # Encode the 'gender' column: 'F' -> 0, 'M' -> 1
    data = data.withColumn("gender_encoded", F.when(F.col("Gender") == "F", 0).otherwise(1))

    # Encode the 'PickUp_Time' column: 'day' -> 0, 'night' -> 1
    data = data.withColumn("pickup_time_encoded", F.when(F.col("PickUp_Time") == "day", 0).otherwise(1))

    # Select relevant columns
    data = data.select("fare", "gender_encoded", "pickup_time_encoded")  # Add other required columns as needed.

    return data
