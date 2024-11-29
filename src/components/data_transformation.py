import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import save_object
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
            categorical_columns = ['car_name', 'brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']

            num_pipeline = Pipeline(
                steps=[("scaler", StandardScaler())]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                    ("scaler", StandardScaler(with_mean=False))  
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            numerical_columns_for_log = ['km_driven', 'seats', 'max_power', 'engine', 'selling_price']
            for column in numerical_columns_for_log:
                train_df[column] = np.log1p(train_df[column])
                test_df[column] = np.log1p(test_df[column])

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "selling_price"
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)

            if hasattr(input_feature_train_arr, 'toarray'):
                input_feature_train_arr = input_feature_train_arr.toarray()

            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            if hasattr(input_feature_test_arr, 'toarray'):
                input_feature_test_arr = input_feature_test_arr.toarray()

            target_feature_train_arr = target_feature_train_df.values.reshape(-1, 1)
            target_feature_test_arr = target_feature_test_df.values.reshape(-1, 1)

            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]

            if input_feature_test_arr.shape[0] == target_feature_test_arr.shape[0]:
                test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]
            else:
                raise CustomException("Error: Shapes of input and target features for test data do not match.", sys)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
