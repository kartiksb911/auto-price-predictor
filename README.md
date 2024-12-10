# AutoPrice Predictor
## Introduction
"This project implements a machine learning model to predict the selling prices of used cars based on various features such as car specifications, mileage, year of manufacture, and more. By utilizing historical data, the model provides accurate price estimates, enabling both buyers and sellers to make well-informed decisions in the used car market."

## Technology Used
* machine learning model-->`XGBoost Regressor`(for price prediction)
* web framework-->`Flask` and `html`(for building the web interface)
* Version Control-->`Git` and `Github`
* code editors-->`jupyter lab `and `Vscode`
* python library-->`numpy`,`Pandas`,`Matplotlib`,`seaborn`,`sklearn`
* Environment: `Anaconda` (for creating isolated environments)
## Features
* Data Preprocessing:This project includes data collection , data analyis,,`feature scaling` ,`binary encoding`and `one hot encoding` for categorical data.
* Model Selection:Various regression models are evaluated,including `Linear regression`,`KNN`,`L1 & L2`,`Random Forest`and `XGBRegressor` to determine the best performing algorithm for price prediction.
* Performance Metrics:use `MAE`,`RMSE`,`R2Square`to ensure reliable prediction.
* Hyperparameters: Tuned hyperparameters such as `learning_rate`, `max_depth`, and `n_estimators` to achieve an `R² Score` of `94%.`
* Web Application: A user friendly web app built with `Flask` allow user to input car details and recieve instantaneous price prediction.
## Usage
#### Clone the Repository

To clone this repository, run the following command:

```
git clone https://github.com/kartiksb911/auto-price-predictor.git
```
#### Create a new environment for the project and activate it.
```
 conda create -p venv python=3.10 -y
 conda activate venv/
```
#### Install all necessary requirements
``` 
pip install -r requirements.txt
```
#### Train the Model
Run the following code to start the data ingestion process and train the model:

``` 
python main.py
```
* After running this code, the model will be trained.
#### Test the Web Application:
Once the model is trained, you can start the Flask web app by running:

```
python app.py
```
Visit http://127.0.0.1:5000 in your browser to interact with the web app.
## 🔗 Check out the deployed web app:
```
https://carprice-1nw9.onrender.com
```
## Final Report
- **R² Score**: 0.94
- **Mean Cross-Validation Score**: 0.93
- **Mean Absolute Error (MAE)**: 61,990.22
- **Root Mean Squared Error (RMSE)**: 88,906.14


## Web App and POSTMAN
![Image Alt](https://github.com/kartiksb911/auto-price-predictor/blob/81168e6485951fa139645be4927fd9d13424f04a/static/Screenshot%20(111).png)
![Image Alt](https://github.com/kartiksb911/auto-price-predictor/blob/81168e6485951fa139645be4927fd9d13424f04a/static/Screenshot%20(112).png)
## EDA
![Image Alt](https://github.com/kartiksb911/auto-price-predictor/blob/81168e6485951fa139645be4927fd9d13424f04a/static/Univariate_Categorcal%20(2).png)
![Image Alt](https://github.com/kartiksb911/auto-price-predictor/blob/81168e6485951fa139645be4927fd9d13424f04a/static/Univariate_Num%20(2).png)
![Image Alt](https://github.com/kartiksb911/auto-price-predictor/blob/81168e6485951fa139645be4927fd9d13424f04a/static/target_vs_continues%20(1).png)
## 🔗 Links
[![Github](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/kartiksb911)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kartik-bhardwaj-07b7282b7/)
 #                                  Thank You
