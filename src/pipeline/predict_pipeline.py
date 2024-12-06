import sys, os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def detect_outliers(self, df, col):
        percentile25 = df[col].quantile(0.25)
        percentile75 = df[col].quantile(0.75)
        iqr = percentile75 - percentile25
        upper_limit = percentile75 + 1.5 * iqr
        lower_limit = percentile25 - 1.5 * iqr
        df[col] = df[col].astype('float64')
        df.loc[(df[col] > upper_limit), col] = upper_limit
        df.loc[(df[col] < lower_limit), col] = lower_limit
        return df
    
    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            skewed_columns = ['km_driven', 'engine', 'max_power', 'seats']  
            for col in skewed_columns:
                features = self.detect_outliers(features, col)
            
            data_scaled = preprocessor.transform(features)
            predictions = model.predict(data_scaled)
            
            return predictions
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 brand: str,
                 model: str,
                 vehicle_age: int,
                 km_driven: int,
                 seller_type: str,
                 fuel_type: str,
                 transmission_type: str,
                 mileage: float,
                 engine: float,
                 max_power: float,
                 seats: int):
        self.brand = brand
        self.model = model
        self.vehicle_age = vehicle_age
        self.km_driven = km_driven
        self.seller_type = seller_type
        self.fuel_type = fuel_type
        self.transmission_type = transmission_type
        self.mileage = mileage
        self.engine = engine
        self.max_power = max_power
        self.seats = seats

    def get_data_as_dataFrame(self):
        try:
            custom_data_input_dict = {
                "brand": [self.brand],
                "model": [self.model],
                "vehicle_age": [self.vehicle_age],
                "km_driven": [self.km_driven],
                "seller_type": [self.seller_type],
                "fuel_type": [self.fuel_type],
                "transmission_type": [self.transmission_type],
                "mileage": [self.mileage],
                "engine": [self.engine],
                "max_power": [self.max_power],
                "seats": [self.seats]
            }
            df = pd.DataFrame(custom_data_input_dict)
            return df
        except Exception as e:
            raise CustomException(e, sys)
