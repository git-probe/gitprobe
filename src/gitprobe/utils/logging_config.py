import logging
import sys


def setup_logging():
    """
    Set up basic logging configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


# You can also get a logger instance to be used across the application
logger = logging.getLogger("gitprobe")
