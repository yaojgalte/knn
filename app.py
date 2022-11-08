import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# title
st.title("Diabetes predictor")
st.text("Made by Yao J. Galteland")
st.subheader("This model will predict if the patient would have diabetes or not (1 = yes, 0 = no). "
             "The accuracy of our model is 97%. "
             "If the intput value is missing, please enter 0.")

# user input
Pregnancies = st.number_input("Pregnancies", step=1, help="indicates the number of pregnancies")
Glucose = st.number_input("Glucose", step=1, help="indicates the plasma glucose concentration")
BloodPressure = st.number_input("Blood Pressure", step=1, help="indicates diastolic blood pressure in mm/Hg")
SkinThickness = st.number_input("Skin Thickness", step=1, help="indicates triceps skinfold thickness in mm")
Insulin = st.number_input("Insulin", step=1, help="indicates insulin in U/mL")
BMI = st.number_input("BMI", format="%0.1f", help="indicates the body mass index in kg/m2")
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", format="%0.3f",
                                           help="indicates the function which scores likelihood of "
                                                "diabetes based on family history")
Age = st.number_input("Age", step=1)

input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

# Use machine learning model to predict diabetes
import pickle
import numpy as np

# load the model from disk
filename = 'finalized_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

def predict_diabetes(input):
    for i in range(1, 7):
        if input[i] == 0:
            input[i] = np.nan
    return loaded_model.predict([input])[0]

# output the prediction
st.subheader("The prediction is {}.".format(predict_diabetes(input)))