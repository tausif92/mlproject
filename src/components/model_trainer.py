from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
# from xgboost import XGBRegressor
from sklearn.metrics import r2_score

import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models
from src.utils import save_object


class ModelTrainerConfig:
    def __init__(self):
        self.trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer(ModelTrainerConfig):
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting train and test data')
            X_train, X_test, y_train, y_test = (
                train_array[:, :-1],
                test_array[:, :-1],
                train_array[:, -1],
                test_array[:, -1]
            )

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    # 'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boost": {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    # 'learning_rate': [.1, .01, .05, .001],
                    # 'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "Ridge": {
                },
                "Lasso": {},
                "Elastic Net": {},
                "SVR": {},
                "K Neighbor": {},
                # "Xgboost": {
                'learning_rate': [.1, .01, .05, .001],
                # 'n_estimators': [8, 16, 32, 64, 128, 256]
                # },
                "Adaboost": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    # 'loss':['linear','square','exponential'],
                    # 'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            models = {
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'Elastic Net': ElasticNet(),
                'SVR': SVR(),
                'K Neighbor': KNeighborsRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'Adaboost': AdaBoostRegressor(),
                'Gradient Boost': GradientBoostingRegressor(),
                # 'Xgboost': XGBRegressor()
            }

            logging.info('Finding the best model')
            model_report = evaluate_models(
                X_train, X_test, y_train, y_test, models, params)

            best_model_score = max(model_report.values())
            dict_keys = list(model_report.keys())
            dict_index = list(model_report.values()).index(best_model_score)
            best_model_name = dict_keys[dict_index]

            best_model_obj = models[best_model_name]
            logging.info(f'Best model: {best_model_name} \t Score: {
                         model_report[best_model_name]}')
            logging.info('Saving model pickle file')
            save_object(
                self.model_trainer_config.trained_model_file_path, best_model_obj)

            predicted = best_model_obj.predict(X_test)
            r2 = r2_score(y_test, predicted)
            return r2

        except Exception as e:
            raise CustomException
