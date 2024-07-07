import yaml
import os
import json
import joblib
import numpy as np
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

params_path = "params.yaml"
schema_path = os.path.join("prediction_service", "schema_in.json")

class NotInRange(Exception):
    def __init__(self, message="Values entered are not in expected range"):
        self.message = message
        super().__init__(self.message)

class NotInCols(Exception):
    def __init__(self, message="Not in cols"):
        self.message = message
        super().__init__(self.message)



def read_params(config_path=params_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def predict(data):
    config = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    model = joblib.load(model_dir_path)
    prediction = model.predict(data).tolist()[0]
    print("Prediction is:", prediction)
    try:
        if 5 <= prediction <= 50:
            return prediction
        else:
            raise NotInRange
    except NotInRange:
        return "Unexpected result"


def get_schema(schema_path=schema_path):
    with open(schema_path) as json_file:
        schema = json.load(json_file)
    return schema

def validate_input(dict_request):
    def _validate_cols(col):
        schema = get_schema()
        actual_cols = schema.keys()
        if col not in actual_cols:
            raise NotInCols

    def _validate_values(col, val):
        schema = get_schema()
        if schema[col].get("min")!= None:
            if not (schema[col]["min"] <= float(dict_request[col]) <= schema[col]["max"]) :
                raise NotInRange

    for col, val in dict_request.items():
        _validate_cols(col)
        _validate_values(col, val)
    
    return True


def form_response(request):
    print("Request Form is:", request.form)
    if validate_input(request.form):
        print("data validated")
        data=CustomData(
            potential=float(request.form.get('potential')),
            crossing=float(request.form.get('crossing')),
            finishing=float(request.form.get('finishing')),
            heading_accuracy=float(request.form.get('heading_accuracy')),
            short_passing=float(request.form.get('short_passing')),
            volleys=float(request.form.get('volleys')),
            dribbling=float(request.form.get('dribbling')),
            curve=float(request.form.get('curve')),
            free_kick_accuracy=float(request.form.get('free_kick_accuracy')),
            long_passing=float(request.form.get('long_passing')),
            ball_control=float(request.form.get('ball_control')),
            acceleration=float(request.form.get('acceleration')),
            sprint_speed=float(request.form.get('sprint_speed')),
            agility=float(request.form.get('agility')),
            reactions=float(request.form.get('reactions')),
            balance=float(request.form.get('balance')),
            shot_power=float(request.form.get('shot_power')),
            jumping=float(request.form.get('jumping')),
            stamina=float(request.form.get('stamina')),
            strength=float(request.form.get('strength')),
            long_shots=float(request.form.get('long_shots')),
            aggression=float(request.form.get('aggression')),
            interceptions=float(request.form.get('interceptions')),
            positioning=float(request.form.get('positioning')),
            vision=float(request.form.get('vision')),
            penalties=float(request.form.get('penalties')),
            marking=float(request.form.get('marking')),
            standing_tackle=float(request.form.get('standing_tackle')),
            sliding_tackle=float(request.form.get('sliding_tackle')),
            gk_diving=float(request.form.get('gk_diving')),
            gk_handling=float(request.form.get('gk_handling')),
            gk_kicking=float(request.form.get('gk_kicking')),
            gk_positioning=float(request.form.get('gk_positioning')),
            gk_reflexes=float(request.form.get('gk_reflexes')),
            
            
            preferred_foot = request.form.get('preferred_foot'),
            attacking_work_rate = request.form.get('attacking_work_rate'),
            defensive_work_rate = request.form.get('defensive_work_rate')
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)
        print(results)
        return results

def api_response(request):
    try:
        if validate_input(request.json):
            
            print("data validated")
            data=CustomData(
            potential=float(request.form.get('potential')),
            crossing=float(request.form.get('crossing')),
            finishing=float(request.form.get('finishing')),
            heading_accuracy=float(request.form.get('heading_accuracy')),
            short_passing=float(request.form.get('short_passing')),
            volleys=float(request.form.get('volleys')),
            dribbling=float(request.form.get('dribbling')),
            curve=float(request.form.get('curve')),
            free_kick_accuracy=float(request.form.get('free_kick_accuracy')),
            long_passing=float(request.form.get('long_passing')),
            ball_control=float(request.form.get('ball_control')),
            acceleration=float(request.form.get('acceleration')),
            sprint_speed=float(request.form.get('sprint_speed')),
            agility=float(request.form.get('agility')),
            reactions=float(request.form.get('reactions')),
            balance=float(request.form.get('balance')),
            shot_power=float(request.form.get('shot_power')),
            jumping=float(request.form.get('jumping')),
            stamina=float(request.form.get('stamina')),
            strength=float(request.form.get('strength')),
            long_shots=float(request.form.get('long_shots')),
            aggression=float(request.form.get('aggression')),
            interceptions=float(request.form.get('interceptions')),
            positioning=float(request.form.get('positioning')),
            vision=float(request.form.get('vision')),
            penalties=float(request.form.get('penalties')),
            marking=float(request.form.get('marking')),
            standing_tackle=float(request.form.get('standing_tackle')),
            sliding_tackle=float(request.form.get('sliding_tackle')),
            gk_diving=float(request.form.get('gk_diving')),
            gk_handling=float(request.form.get('gk_handling')),
            gk_kicking=float(request.form.get('gk_kicking')),
            gk_positioning=float(request.form.get('gk_positioning')),
            gk_reflexes=float(request.form.get('gk_reflexes')),
            
            
            preferred_foot = request.form.get('preferred_foot'),
            attacking_work_rate = request.form.get('attacking_work_rate'),
            defensive_work_rate = request.form.get('defensive_work_rate')
            )
        
            final_new_data=data.get_data_as_dataframe()
            predict_pipeline=PredictPipeline()
            pred=predict_pipeline.predict(final_new_data)

            results=round(pred[0],2)
            print(results)
            response = {"response": results}
            return response
            
    except NotInRange as e:
        response = {"the_exected_range": get_schema(), "response": str(e) }
        return response

    except NotInCols as e:
        response = {"the_exected_cols": get_schema().keys(), "response": str(e) }
        return response


    except Exception as e:
        response = {"response": str(e) }
        return response