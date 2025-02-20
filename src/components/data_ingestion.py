import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import sqlite3

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionconfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')
    
## create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()
    
    def initiate_data_ingestion(self):
        logging.info('Data Ingestion methods Starts')
        try:
            cnx = sqlite3.connect(r'data/database.sqlite')
            df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)
            logging.info('Dataset read as pandas Dataframe')
            
            ## Unwanted columns can be deleted here
            df = df.drop(['id', 'date','player_fifa_api_id', 'player_api_id'], axis=1)
            
            # Convert all None values to np.nan
            df = df.where(pd.notnull(df), np.nan)
            df = df.dropna(subset=['overall_rating'])
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('Train test split')
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
  
            
        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_obj_file_path, = data_transformation.initaite_data_transformation(train_data_path,test_data_path)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr, test_arr)