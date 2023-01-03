## 1-Import libraries
from numpy import nan
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

## 2-Import dataset
df = pd.read_csv('diabetes.csv')
# display all columns
pd.set_option("display.max_columns", None)

## 3-Exploratory data analysis
# Too many unrealistic zeros, change zero to nan
for col in df:
    if col == 'Pregnancies' or col == 'Outcome':
        continue
    df.loc[df[df[col] == 0].index, col] = nan

## 4- use KNNImputer to impute missing values in step 6

## 5-Set X and y variables
# split into input and output elements
data = df.values
ix = [i for i in range(data.shape[1]) if i != 8]
X, y = data[:, ix], data[:, 8]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10, shuffle=True)

## 6-Set algorithm
# define modeling pipeline
# use KNNImputer to impute missing values
# set algorithm by random forest algorithm
pipeline = Pipeline(steps=[('i', KNNImputer(n_neighbors=21)), ('m', RandomForestClassifier())])
# fit the model
pipeline.fit(X_train, y_train)

## 7-Evaluate
model_predict = pipeline.predict(X_test)
print(pipeline.score(X_test, y_test))
print(confusion_matrix(y_test, model_predict))
print(classification_report(y_test, model_predict))




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

def predict_diabetes(input):
    for i in range(1, 7):
        if input[i] == 0:
            input[i] = np.nan
    return pipeline.predict([input])[0]

# output the prediction
st.subheader("The prediction is {}.".format(predict_diabetes(input)))