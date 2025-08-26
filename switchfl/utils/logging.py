import logging


class CustomFormatter(logging.Formatter):
    """class copied from: https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output"""

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def format_logger(logger: logging.Logger) -> logging.Logger:
    """Format a logger with a custom formatter.

    Args:
        logger (logging.Logger): The logger to format.

    Returns:
        logging.Logger: The formatted logger.
    """
    logger.propagate = False  # Prevent duplicate logs
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)
    return logger
