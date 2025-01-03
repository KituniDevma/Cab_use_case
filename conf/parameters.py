parameters = {
    # Paths
    "paths": {
        "data_path": "../Cab_use_case/src/data/Colombo-Cab-data-2.csv",
        "dictionary_path": "../Cab_use_Case/src/data/Colombo-Cab-data-dictionary.csv",
        "output_path": "/workspaces/Cab_use_case/src/data/output.csv/part-00000-727c7fda-0f04-459c-8a39-dd44229c4c83-c000.csv",
    },

    # Spark parameters
    "spark": {
        "app_name": "MySparkApp",
        "master": "local[2]",  # Use local mode with 2 cores for testing
        "config": {
            "spark.executor.memory": "2g",
            "spark.driver.memory": "2g",
        },
    },

    # XGBoost model parameters
    "xgboost": {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "learning_rate": 0.1,  # Also known as 'eta'
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": 1,
        "eval_metric": "rmse",
    },

    # Train-test split parameters
    "train_test_split": {
        "target_column": 'Fare',
        "test_size": 0.2,
        "random_state": 10,
    }
}
