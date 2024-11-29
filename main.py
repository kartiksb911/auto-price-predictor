from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

if __name__ == "__main__":
    try:
        ingestion_obj = DataIngestion()
        train_path, test_path = ingestion_obj.initiate_data_ingestion()

        transformation_obj = DataTransformation()
        train_arr, test_arr,_= transformation_obj.initiate_data_transformation(train_path, test_path)
        
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
