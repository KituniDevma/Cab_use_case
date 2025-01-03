import logging
import logging.config
import yaml

def setup_logging(config_file="logging_config.yaml"):
    """
    Set up logging configuration from a YAML file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        None
    """
    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
            logging.config.dictConfig(config)
        print("Logging configuration loaded successfully.")
    except Exception as e:
        print(f"Error loading logging configuration: {e}")

def get_logger(logger_name):
    """
    Get a logger by name.

    Args:
        logger_name (str): Name of the logger.

    Returns:
        Logger: Configured logger instance.
    """
    return logging.getLogger(logger_name)

def log_info(message):
    """
    Log an informational message.

    Args:
        message (str): The message to log.

    Returns:
        None
    """
    info_logger = get_logger("info_logger")
    info_logger.info(message)

def log_warning(message):
    """
    Log a warning message.

    Args:
        message (str): The message to log.

    Returns:
        None
    """
    warning_logger = get_logger("warning_logger")
    warning_logger.warning(message)

