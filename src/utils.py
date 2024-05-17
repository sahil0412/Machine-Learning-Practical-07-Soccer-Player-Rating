import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train,y_train,X_test,y_test,models, param_grids):
    try:
        
        report = {}
        best_model = None
        best_score = -float('inf')
        best_params = None
        for name, model in models.items():
            param_grid = param_grids[name]
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
            grid_search.fit(X_train, y_train)
            best_estimator = grid_search.best_estimator_
            
            # Predict Testing data
            y_test_pred = best_estimator.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[name] = {
                'best_params': grid_search.best_params_,
                'test_r2_score': test_model_score
            }
            # Check if this is the best model so far
            if test_model_score > best_score:
                best_model = best_estimator
                best_score = test_model_score
                best_params = grid_search.best_params_
        return report, best_model, best_params
        
        
        # model.fit(X_train, y_train)
        # print('Training Score : ', model.score(X_train, y_train))
        # print('Testing Score  : ', model.score(X_test, y_test))
        # # Predict Testing data
        # test_accuracy = model.score(X_test, y_test)
        # y_test_pred =model.predict(X_test)
        # cs = confusion_matrix(y_test, y_test_pred)
        # return test_accuracy, cs, model
    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)