import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import save_object
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders.binary import BinaryEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

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

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
            categorical_columns = ['seller_type','fuel_type','transmission_type']
            binary_columns = ['brand', 'model']

            num_pipeline = Pipeline(
                steps=[("scaler", StandardScaler())]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                    ("scaler", StandardScaler(with_mean=False))  
                ]
            )

            bin_pipeline=Pipeline(
                steps=[("binary",BinaryEncoder())]
            )
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                    ("bin_pipeline",bin_pipeline,binary_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            skwed_data = ['km_driven', 'engine', 'max_power', 'seats', 'selling_price']
            for col in skwed_data:
                train_df = self.detect_outliers(train_df, col)
                test_df = self.detect_outliers(test_df, col)

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
                self.data_transformation_config.preprocessor_obj_file_path,
                
            )

        except Exception as e:
            raise CustomException(e, sys)
