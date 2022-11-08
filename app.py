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

Pregnancies = st.number_input("Gravidity", step=1, help="the number of times that a woman has been pregnant")
Glucose = st.number_input("Glucose", step=1)
BloodPressure = st.number_input("BloodPressure", step=1)
SkinThickness = st.number_input("SkinThickness", step=1)
Insulin = st.number_input("Insulin", step=1)
BMI = st.number_input("BMI")
DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction")
Age = st.number_input("Age", step=1)

input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

# knn imputation strategy and
# prediction for the diabetes


## 1-Import libraries

from numpy import nan
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import ExtraTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, cross_val_predict
from sklearn.preprocessing import RobustScaler, PowerTransformer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import pickle

## 2-Import dataset
df = pd.read_csv('diabetes.csv')
# display all columns
pd.set_option("display.max_columns", None)

# Interestingly many columns contain zeroes as can be seen by the min column; realistically only Pregnancies
# should be able to be set to an absolute zero
# change zero to nan
for col in df:
    if col == 'Pregnancies' or col == 'Outcome':
        continue
    df.loc[df[df[col] == 0].index, col] = nan

## 5-Set X and y variables
data = df.values
ix = [i for i in range(data.shape[1]) if i != 8]
X, y = data[:, ix], data[:, 8]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10, shuffle=True)


## 6-Set algorithm
# Here I have included the KNNImputer to deal with the NaNs in the dataset
# There are a lot of NaNs in the Insulin column, however I did experiment with just dropping the NaNs and the
# results were not as good as with KNN Imputing

def make_clf_pipeline(alg):
    return Pipeline([
        ('scaler', RobustScaler()),
        ('imputer', KNNImputer(n_neighbors=5, missing_values=np.nan)),
        ('clf', alg)
    ])

clf = make_clf_pipeline(ExtraTreeClassifier())
cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

for train, test in cv.split(X, y):
    model = make_clf_pipeline(RandomForestClassifier(random_state=42, n_jobs=-1)).fit(X[train], y[train])
    y_score = model.predict_proba(X[test])


## 7-Evaluate
y_pred = cross_val_predict(make_clf_pipeline(RandomForestClassifier(random_state=42, n_jobs=-1)),
                           cv=cv, X=X, y=y, n_jobs=-1)


# predict
for i in range(1, 7):
    if input[i] == 0:
        input[i] = np.nan
output = 1
st.subheader("The prediction is {}.".format(output))

st.subheader("The prediction is {}.".format(model.predict([input])[0]))


