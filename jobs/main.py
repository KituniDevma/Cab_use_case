from pyspark.sql import SparkSession
from conf.config import init_spark, load_csv_data
from pipeline.data_preprocessing import preprocess
from conf.parameters import parameters

def main():
    # Initialize Spark session
    spark = init_spark(
        app_name=parameters['spark']['app_name'],
        master=parameters['spark']['master'],
        executor_memory=parameters['spark']['config']['spark.executor.memory'],
        driver_memory=parameters['spark']['config']['spark.driver.memory']
    )

    # Load the data
    input_path = parameters['paths']['data_path']
    data = load_csv_data(spark, input_path)

    # Preprocess the data
    processed_data = preprocess(data)

    # Save the processed data
    output_path = parameters['paths']['output_path']
    processed_data.write.csv(output_path, header=True, mode='overwrite')
    print(f"Processed data saved to {output_path}")

    # Stop the Spark session
    spark.stop()

if __name__ == '__main__':
    main()


