from src.logger import logging
from src.exception import CustomException
import sys
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.utils import save_object


class DataTransformationConfig:
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join(
            'artifacts', 'preprocessor.pkl')


class DataTransformation(DataTransformationConfig):
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            num_columns = ['reading_score', 'writing_score']
            cat_columns = ['gender', 'race_ethnicity',
                           'parental_level_of_education', 'lunch', 'test_preparation_course']
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_columns),
                    ('cat_pipeline', cat_pipeline, cat_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info('Reading train and test data')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Obtaining preprocessing object')
            preprocessor_obj = self.get_data_transformation_obj()

            logging.info('Setting input and target features')
            target_column_name = 'math_score'
            input_feature_train_df = train_df.drop(
                columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(
                columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('Applying preprocessor object on train and test data')
            input_feature_train_arr = preprocessor_obj.fit_transform(
                input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(
                input_feature_test_df)

            logging.info('Getting train and test array')
            train_arr = np.c_[input_feature_train_arr,
                              np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,
                             np.array(target_feature_test_df)]

            logging.info('Saving preprocessor object')
            save_object(
                file_path=self.data_tranformation_config.preprocessor_obj_file_path, obj=preprocessor_obj)

            return train_arr, test_arr, self.data_tranformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
