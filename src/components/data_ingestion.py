import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


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
            logging.info("Reading dataset started")
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Reading dataset as dataframe completed")

            # os.makedirs(self.ingestion_config.raw_data_path, exist_ok=True)
            logging.info(f'Saving raw data in path: {
                         self.ingestion_config.raw_data_path}')
            df.to_csv(self.ingestion_config.raw_data_path,
                      index=False, header=True)
            logging.info('Train test split started')
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42)
            logging.info(f'Saving train data in path: {
                         self.ingestion_config.train_data_path}')
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True)
            logging.info(f'Saving test data in path: {
                         self.ingestion_config.test_data_path}')
            test_set.to_csv(self.ingestion_config.test_data_path,
                            index=False, header=True)

            logging.info('Ingestion of data is completed')

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(sys, e)


if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data_path, test_data_path)

    trainer = ModelTrainer()
    r2_accuracy = trainer.initiate_model_trainer(train_arr, test_arr)
    print(r2_accuracy)
