import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split

def train_test_split_data(data,test_percentage=0.2):
    try:
        train , test =train_test_split(data,test_size=test_percentage)
        return train,test
    except Exception as e:
        raise CustomException(e,sys)
    


   