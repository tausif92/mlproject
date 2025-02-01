from src.exception import CustomException
import os
import pickle
import sys
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, X_test, y_train, y_test, models, params):
    try:
        model_r2_score = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)
            # model.fit(X_train, y_train)

            # Make predictions
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_test_pred = gs.predict(X_test)

            # Evaluate r2 score
            r2_score_test = r2_score(y_test, y_test_pred)

            # Add model result to dictionary
            model_r2_score[list(models.keys())[i]] = r2_score_test

        return model_r2_score

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
