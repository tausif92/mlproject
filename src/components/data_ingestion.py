import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split


class DataIngestionConfig:
    def __init__(self):
        self.train_data_path = os.path.join('artifacts', 'train.csv')
        self.test_data_path = os.path.join('artifacts', 'test.csv')
        self.raw_data_path = os.path.join('artifacts', 'raw.csv')
        os.makedirs('artifacts', exist_ok=True)


class DataIngestion(DataIngestionConfig):
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")

        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Reading dataset as dataframe completed")

            # os.makedirs(self.ingestion_config.raw_data_path, exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,
                      index=False, header=True)

            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,
                             index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,
                            index=False, header=True)

            logging.info('Ingestion of data is completed')

        except Exception as e:
            raise CustomException(sys, e)


if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()
