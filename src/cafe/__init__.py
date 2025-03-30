import logging
from . import schedule

# Set up default logging configuration
logging.basicConfig(
    level=logging.INFO,  # Default logging level
    # format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    format="%(levelname)s: %(message)s",  # Log format
)