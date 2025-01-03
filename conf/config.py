import logging
from pyspark.sql import SparkSession

# Initialize logger
logger = logging.getLogger("SparkAppLogger")

def init_spark(app_name="SparkApp", master="local[*]", executor_memory="2g", driver_memory="2g"):
    """
    Initializes a Spark session with configurations.

    Args:
        app_name (str): Name of the Spark application.
        master (str): Master URL for the cluster.
        executor_memory (str): Memory allocated to the executor.
        driver_memory (str): Memory allocated to the driver.

    Returns:
        SparkSession: An initialized Spark session.
    """
    try:
        spark = SparkSession.builder \
            .appName(app_name) \
            .master(master) \
            .config("spark.executor.memory", executor_memory) \
            .config("spark.driver.memory", driver_memory) \
            .getOrCreate()
        logger.info(f"Spark session initialized successfully: app_name={app_name}, master={master}")
        return spark
    except Exception as e:
        logger.error(f"Failed to initialize Spark session: {e}")
        raise

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

