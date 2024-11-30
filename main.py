from src.exception import CustomException
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    try:
        ingestion_obj = DataIngestion()
        train_path, test_path = ingestion_obj.initiate_data_ingestion()

        transformation_obj = DataTransformation()
        train_arr, test_arr,_= transformation_obj.initiate_data_transformation(train_path, test_path)

        model_trainer=ModelTrainer()
        model_trainer.initiate_model_trainer(train_arr,test_arr)
        
        
    except Exception as e:
        raise CustomException(e,sys)
