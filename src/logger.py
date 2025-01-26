from pathlib import Path
import os
import logging
from datetime import datetime

# Set logs path
LOG_FILE_NAME = f"{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log"
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR_PATH = BASE_DIR / 'logs'

# Create log file
os.makedirs(LOG_DIR_PATH, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR_PATH, LOG_FILE_NAME)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
