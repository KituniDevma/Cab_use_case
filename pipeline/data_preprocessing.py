from pyspark.sql import functions as F
import logging

# Initialize logger
logger = logging.getLogger("PreprocessingLogger")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def preprocess(data):
    """
    Preprocess the data: create new feature 'Fare', and encode categorical variables.

    Args:
        data: Spark DataFrame to preprocess.

    Returns:
        DataFrame: Preprocessed Spark DataFrame with all original columns and new features.
    """
    try:
        logger.info("Starting preprocessing of the data.")
        
        # Create the 'Fare' column
        logger.info("Creating the 'Fare' column by subtracting 'Tip' from 'Total_Amount'.")
        data = data.withColumn("Fare", F.col("Total_Amount") - F.col("Tip"))
        logger.debug("Fare column added successfully.")

        # Encode the 'Gender' column
        logger.info("Encoding the 'Gender' column: 'F' -> 0, 'M' -> 1.")
        data = data.withColumn("Gender_Encoded", F.when(F.col("Gender") == "F", 0).otherwise(1))
        logger.debug("Gender column encoded successfully.")

        # Encode the 'PickUp_Time' column
        logger.info("Encoding the 'PickUp_Time' column: 'day' -> 0, 'night' -> 1.")
        data = data.withColumn("PickUp_Time_Encoded", F.when(F.col("PickUp_Time") == "day", 0).otherwise(1))
        logger.debug("PickUp_Time column encoded successfully.")

        # Convert the 'Date' column to a timestamp or date format
        logger.info("Converting the 'Date' column to a date format (yyyy-MM-dd).")
        data = data.withColumn('Date', F.to_date(data['Date'], 'yyyy-MM-dd'))
        logger.debug("Date column converted successfully.")

        # Drop the original columns used for new feature creation
        logger.info("Dropping original columns: 'Gender', 'PickUp_Time', 'Total_Amount', and 'Date'.")
        data = data.drop("Gender", "PickUp_Time", "Total_Amount", "Date")
        logger.info("Data preprocessing completed successfully.")
        
        return data

    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        raise

