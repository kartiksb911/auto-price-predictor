import os,sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import train_test_split_data

@dataclass
class DataIngestionConfig:
        train_data_path=os.path.join("artifacts","train.csv")
        test_data_path=os.path.join("artifacts","test.csv")
        raw_data_path=os.path.join("artifacts","raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            data_file_path = "notebook/data/cardata.csv"
            if not os.path.exists(data_file_path):
                raise FileNotFoundError(f"CSV file not found at {data_file_path}")
    
            df = pd.read_csv(data_file_path)
            logging.info("reading the csv data")

            df=df.drop(df.columns[0],axis=1)
            logging.info("drop the first column") 

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("splitting the data")
            
            train_data,test_data=train_test_split_data(df,0.2)
            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )        
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
     obj=DataIngestion()
     train_path,test_path=obj.initiate_data_ingestion()
