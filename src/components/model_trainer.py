import os
import numpy as np
import sys
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import cross_val_score
from src.utils import  save_object
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            xgb_model = XGBRegressor(learning_rate= 0.1, max_depth= 7, n_estimators=200)
            xgb_model.fit(x_train,y_train)
            y_pred=xgb_model.predict(x_test)
            mse = mean_squared_error(y_test, y_pred)  
            r2 = r2_score(y_test, y_pred)
            mae=mean_absolute_error(y_test,y_pred)
            rmse = np.sqrt(mse)  

            print(f"mse is :{mse}")
            print(f"r2score is :{r2}")
            print(f"mae is :{mae}")
            print(f"rmse is :{rmse}")
            logging.info(f"mse is :{mse}")
            logging.info(f"r2score is :{r2}")            
            logging.info(f"mae is :{mae}")
            logging.info(f"rmse is :{rmse}")

            cross_val_scores = cross_val_score(xgb_model, x_train, y_train, cv=5)
            print("Cross-validation scores:", cross_val_scores)
            print("Mean cross-validation score:", cross_val_scores.mean())
            logging.info(f"Cross-validation scores: {cross_val_scores}")
            logging.info(f"Mean cross-validation score: {cross_val_scores.mean()}")

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=xgb_model)
           
        except Exception as e:
            raise CustomException(e, sys)