from pyspark.sql import SparkSession

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
    spark = SparkSession.builder \
        .appName(app_name) \
        .master(master) \
        .config("spark.executor.memory", executor_memory) \
        .config("spark.driver.memory", driver_memory) \
        .getOrCreate()
    return spark

def load_csv_data(spark, file_path):
    """
    Loads a CSV file into a Spark DataFrame.

    Args:
        spark (SparkSession): Active Spark session.
        file_path (str): Path to the CSV file.

    Returns:
        DataFrame: Loaded Spark DataFrame.
    """
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    print(f"Data loaded successfully from {file_path}")
    return df
