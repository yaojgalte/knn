import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#df = pd.read_csv("diabetes.csv")

# title
st.title("Diabetes predictor")
st.subheader("This model will predict if a person would have diabetes or not. "
             "If the intput value is missing, please enter 0.")

Pregnancies = st.text_input("Gravidity (the number of times that a woman has been pregnant).")
Glucose = st.text_input("Glucose")
BloodPressure = st.text_input("BloodPressure")
SkinThickness = st.text_input("SkinThickness")
Insulin = st.text_input("Insulin")
BMI = st.text_input("BMI")
DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction")
Age = st.text_input("Age")

input = pd.Series([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
pd.to_numeric(input) # convert everything to float values


