import logging
import sys

COLORS = {
    "DEBUG":    "\033[94m",  
    "INFO":     "\033[92m",  
    "WARNING":  "\033[93m",  
    "ERROR":    "\033[91m",  
}
RESET = "\033[0m"

class ColorFormatter(logging.Formatter):
    def format(self, record):
        levelname = record.levelname
        if levelname in COLORS:
            record.levelname = COLORS[levelname] + levelname + RESET
        return super().format(record)


def setup_logger():
    LOG_FORMAT = "%(asctime)s | %(levelname)-16s | %(message)s"
    DATE_FORMAT = "%H:%M:%S"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = [handler]

    return logger
