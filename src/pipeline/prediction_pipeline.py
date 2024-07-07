import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')
            
            preprocessing_data=load_object(preprocessor_path)
            model=load_object(model_path)
            
            categorical_features=['preferred_foot', 'attacking_work_rate', 'defensive_work_rate']
            numerical_features=['potential', 'crossing',
                            'finishing', 'heading_accuracy', 'short_passing', 'volleys',
                            'dribbling', 'curve', 'free_kick_accuracy', 'long_passing',
                            'ball_control', 'acceleration', 'sprint_speed', 'agility', 'reactions',
                            'balance', 'shot_power', 'jumping', 'stamina', 'strength', 'long_shots',
                            'aggression', 'interceptions', 'positioning', 'vision', 'penalties',
                            'marking', 'standing_tackle', 'sliding_tackle', 'gk_diving',
                            'gk_handling', 'gk_kicking', 'gk_positioning', 'gk_reflexes']
            input_feature_test_arr = pd.DataFrame()
            l_imputer = preprocessing_data['categorical_imputer']
            input_feature_test_arr[categorical_features] = l_imputer.transform(features[categorical_features])
            # Transform the categorical columns in the test data
            for column in categorical_features:
                l_encoder = preprocessing_data[f'{column}_label_encoder']
                input_feature_test_arr[column] = input_feature_test_arr[column].map(lambda s: -1 if s not in l_encoder.classes_ else l_encoder.transform([s])[0])

            imputer = preprocessing_data['numerical_imputer']
            scaler = preprocessing_data['numerical_scaler']

            input_feature_test_arr[numerical_features] = imputer.transform(features[numerical_features])
            input_feature_test_arr[numerical_features] = scaler.transform(input_feature_test_arr[numerical_features])
            
            
            # data_scaled=preprocessor.transform(features)

            pred=model.predict(input_feature_test_arr)
            print("Prediction is:", pred)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
        
class CustomData:
    def __init__(self,
                 potential:float,
                 crossing:float,
                 finishing:float,
                 heading_accuracy:float,
                 short_passing:float,
                 volleys:float,
                 dribbling:float,
                 curve:float,
                 free_kick_accuracy:float,
                 long_passing:float,
                 ball_control:float,
                 acceleration:float,
                 sprint_speed:float,
                 agility:float,
                 reactions:float,
                 balance:float,
                 shot_power:float,
                 jumping:float,
                 stamina:float,
                 strength:float,
                 long_shots:float,
                 aggression:float,
                 interceptions:float,
                 positioning:float,
                 vision:float,
                 penalties:float,
                 marking:float,
                 standing_tackle:float,
                 sliding_tackle:float,
                 gk_diving:float,
                 gk_handling:float,
                 gk_kicking:float,
                 gk_positioning:float,
                 gk_reflexes:float,
                 
                 preferred_foot:str,
                 attacking_work_rate:str,
                 defensive_work_rate:str
                 ):
        self.potential=potential
        self.crossing=crossing
        self.finishing=finishing
        self.heading_accuracy=heading_accuracy
        self.short_passing = short_passing
        self.volleys = volleys
        self.dribbling = dribbling
        self.curve = curve
        self.free_kick_accuracy = free_kick_accuracy
        self.long_passing = long_passing
        self.ball_control = ball_control
        self.acceleration = acceleration
        self.sprint_speed = sprint_speed
        self.agility = agility
        self.reactions = reactions
        self.balance = balance
        self.shot_power = shot_power
        self.jumping = jumping
        self.stamina = stamina
        self.strength = strength
        self.long_shots = long_shots
        self.aggression = aggression
        self.interceptions = interceptions
        self.positioning = positioning
        self.vision = vision
        self.penalties = penalties
        self.marking = marking
        self.standing_tackle = standing_tackle
        self.sliding_tackle = sliding_tackle
        self.gk_diving = gk_diving
        self.gk_handling = gk_handling
        self.gk_kicking = gk_kicking
        self.gk_positioning = gk_positioning
        self.gk_reflexes = gk_reflexes
        
        self.preferred_foot = preferred_foot
        self.attacking_work_rate = attacking_work_rate
        self.defensive_work_rate = defensive_work_rate
        

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'potential':[self.potential],
                'crossing':[self.crossing],
                'finishing':[self.finishing],
                'heading_accuracy':[self.heading_accuracy],
                'short_passing':[self.short_passing],
                'volleys':[self.volleys],
                'dribbling':[self.dribbling],
                'curve':[self.curve],
                'free_kick_accuracy':[self.free_kick_accuracy],
                "long_passing":[self.long_passing],
                "ball_control":[self.ball_control],
                "acceleration":[self.acceleration],
                "sprint_speed":[self.sprint_speed],
                "agility":[self.agility],
                "reactions":[self.reactions],
                "balance":[self.balance],
                "shot_power":[self.shot_power],
                "jumping":[self.jumping],
                "stamina":[self.stamina],
                "strength":[self.strength],
                "long_shots":[self.long_shots],
                "aggression":[self.aggression],
                "interceptions":[self.interceptions],
                "positioning":[self.positioning],
                "vision":[self.vision],
                "penalties":[self.penalties],
                "marking":[self.marking],
                "standing_tackle":[self.standing_tackle],
                "sliding_tackle":[self.sliding_tackle],
                "gk_diving":[self.gk_diving],
                "gk_handling":[self.gk_handling],
                "gk_kicking":[self.gk_kicking],
                "gk_positioning":[self.gk_positioning],
                "gk_reflexes":[self.gk_reflexes],
                
                
                "preferred_foot":[self.preferred_foot],
                "attacking_work_rate":[self.attacking_work_rate],
                "defensive_work_rate":[self.defensive_work_rate],
                
                
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)