import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
import dill

def train_test_split_data(data,test_percentage=0.2):
    try:
        train , test =train_test_split(data,test_size=test_percentage)
        return train,test
    except Exception as e:
        raise CustomException(e,sys)
    


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)     