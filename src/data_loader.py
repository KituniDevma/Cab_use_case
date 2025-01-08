import logging

logger = logging.getLogger("DataLoaderLogger")

def load_csv_data(spark, file_path):
    """
    Loads a CSV file into a Spark DataFrame.

    Args:
        spark (SparkSession): Active Spark session.
        file_path (str): Path to the CSV file.

    Returns:
        DataFrame: Loaded Spark DataFrame.
    """
    try:
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        logger.info(f"Data loaded successfully from {file_path}")
        logger.debug(f"Schema of loaded data: {df.schema}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        raise

