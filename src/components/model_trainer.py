# Basic Import
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            # Define hyperparameter grids for each model
            param_grids = {
                'LinearRegression': {},  # No hyperparameters to tune for Linear Regression
                'DecisionTree': {'max_depth': [10, 13, 15]},  # Hyperparameters to tune for Decision Tree
                'RandomForestRegressor': {'max_depth': [8, 10, 12], 'max_features': ['sqrt', 'log2'], 'n_estimators': [100]},
                'XGBRegressor': {'learning_rate': [0.1, 0.3, 0.5], 'max_depth': [5, 7, 9], 'n_estimators': [100]}
            }

            # Update models dictionary with base models
            models = {
                'LinearRegression': LinearRegression(),
                'DecisionTree': DecisionTreeRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'XGBRegressor': XGBRegressor()
            }


            model_report, best_model, best_params = evaluate_model(X_train, y_train, X_test, y_test, models, param_grids)
            
            # model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_name = max(model_report, key=lambda k: model_report[k]['test_r2_score'])
            best_model_score = model_report[best_model_name]['test_r2_score']

            # best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print(f'Best Parameters: {best_params}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            logging.info(f'Best Parameters: {best_params}')
            
            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)