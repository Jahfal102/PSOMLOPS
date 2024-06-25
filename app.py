import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time

from matplotlib import style
style.use("seaborn")
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')
def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

set_background('download(!).png')

st.write("## Calories burned Prediction")
st.write("In this WebApp you will be able to observe your predicted calories burned in your body. Only thing you have to do is pass your parameters such as `Age`, `Gender`, `BMI`, etc. into this WebApp and then you will be able to see the predicted value of kilocalories that burned in your body.")

st.sidebar.header("User Input Parameters : ")

def user_input_features():
    age = st.sidebar.slider("Age: ", 10, 100, 30)
    bmi = st.sidebar.slider("BMI: ", 15, 40, 20)
    duration = st.sidebar.slider("Duration (min): ", 0, 35, 15)
    heart_rate = st.sidebar.slider("Heart Rate: ", 60, 130, 80)
    body_temp = st.sidebar.slider("Body Temperature (C): ", 36, 42, 38)
    gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"))

    gender = 1 if gender_button == "Male" else 0

    data = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender": gender
    }

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.write("---")
st.header("Your Parameters : ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)
st.write(df)

@st.cache
def load_data():
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
    exercise_df = exercise.merge(calories, on="User_ID")
    exercise_df.drop(columns="User_ID", inplace=True)
    return exercise_df

exercise_df = load_data()

exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

for data in [exercise_train_data, exercise_test_data]:
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"], 2)

exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]

X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)

# Ensure df has the same structure as X_train
df = pd.get_dummies(df, drop_first=True)
missing_cols = set(X_train.columns) - set(df.columns)
for col in missing_cols:
    df[col] = 0
df = df[X_train.columns]

prediction = random_reg.predict(df)

st.write("---")
st.header("Prediction : ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

st.write(round(prediction[0], 2), "   **kilocalories**")

st.write("---")
st.header("Similar Results : ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

range = [prediction[0] - 10, prediction[0] + 10]
ds = exercise_df[(exercise_df["Calories"] >= range[0]) & (exercise_df["Calories"] <= range[-1])]
st.write(ds.sample(5))

st.write("---")
st.header("General Information : ")

boolean_age = (exercise_df["Age"] < df['Age'][0]).tolist()
boolean_duration = (exercise_df["Duration"] < df['Duration'][0]).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < df['Body_Temp'][0]).tolist()
boolean_heart_rate = (exercise_df["Heart_Rate"] < df['Heart_Rate'][0]).tolist()

st.write("You are older than %", round(sum(boolean_age) / len(boolean_age), 2) * 100, "of other people.")
st.write("You had higher exercise duration than %", round(sum(boolean_duration) / len(boolean_duration), 2) * 100, "of other people.")
st.write("You had more heart rate than %", round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100, "of other people during exercise.")
st.write("You had higher body temperature than %", round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100, "of other people during exercise.")
