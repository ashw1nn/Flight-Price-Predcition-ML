import sys, os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        airline: str,
        source: str,
        destination: str,
        additional_info: str,
        duration: int,
        total_stops: float,
        date: int,
        month: int,
        year: int,
        arrival_hour: int,
        arrival_minute: int,
        dep_hour: int,
        dep_minute: int,):

        self.airline = airline

        self.source = source

        self.destination = destination

        self.additional_info = additional_info

        self.duration = duration

        self.total_stops = total_stops

        self.date = date

        self.month = month

        self.year = year

        self.arrival_hour = arrival_hour

        self.arrival_minute = arrival_minute

        self.dep_hour = dep_hour

        self.dep_minute = dep_minute

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Airline": [self.airline],
                "Source": [self.source],
                "Destination": [self.destination],
                "Additional_Info": [self.additional_info],
                "Duration": [self.duration],
                "Total_Stops": [self.total_stops],
                "Date": [self.date],
                "Month": [self.month],
                "Year": [self.year],
                "Arrival_Hour": [self.arrival_hour],
                "Arrival_Minute": [self.arrival_minute],
                "Dep_Hour": [self.dep_hour],
                "Dep_Minute": [self.dep_minute],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

