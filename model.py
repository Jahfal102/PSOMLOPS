import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

#loading the data from csv file to a Pandas DataFrame
calories = pd.read_csv('calories.csv')
calories.head()

exercise_data = pd.read_csv('exercise.csv')
exercise_data.head()

calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)
calories_data.shape

calories_data.isnull().sum()
calories_data.describe()

calories_data.replace({"Gender":{'male':0,'female':1}}, inplace=True)

calories_data.corr()

X = calories_data.drop(columns=['User_ID','Calories'], axis=1)
Y = calories_data['Calories']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# loading the model
model = XGBRegressor()
model.fit(X_train, Y_train)

model.predict(X_test)
test_data_prediction

mae = metrics.mean_absolute_error(Y_test, test_data_prediction)
