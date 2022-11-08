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

Pregnancies = st.number_input("Gravidity (the number of times that a woman has been pregnant).",step=1, help="help")

