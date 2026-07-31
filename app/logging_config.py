import logging


logger = logging.getLogger("credit-risk-api")


def setup_logging():

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s "
        "%(levelname)s "
        "%(name)s "
        "%(message)s"
    )

    # Avoid duplicate handlers during reload/tests
    if logger.handlers:
        return

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        console_handler
    )