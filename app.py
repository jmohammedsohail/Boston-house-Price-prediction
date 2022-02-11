import pickle
import pandas as pd
import numpy as np
import streamlit as st
import requests
import sklearn

# load the model from disk
model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))


# # Loads the Boston House Price Dataset
X= pd.read_csv("Boston.csv")
X=X.drop('Price',axis=1) 
X.min()
X.info()
st.write("""
# Boston House Price Prediction App
""")
st.caption('                   By Mohammed Sohail')
st.markdown('This app predicts the **Boston House Price**!')
st.write('---')

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')
float(X.CRIM.min())
X.CRIM.max()
def user_input_features():
    CRIM = st.sidebar.slider('CRIM', min_value=float(X.CRIM.min()), max_value=float(X.CRIM.max()),X.CRIM.mean())
    ZN = st.sidebar.slider('ZN',min_value=float(X.ZN.min()), max_value=float(X.ZN.max()), step=float(X.ZN.mean()))
    INDUS = st.sidebar.slider('INDUS',min_value=float(X.INDUS.min()), max_value=float(X.INDUS.max()), step=float(X.INDUS.mean()))
    CHAS = st.sidebar.slider('CHAS', min_value=float(X.CHAS.min()),max_value= float(X.CHAS.max()), step=float(X.CHAS.mean()))
    NOX = st.sidebar.slider('NOX', min_value=float(X.NOX.min()), max_value=float(X.NOX.max()),step=float( X.NOX.mean()))
    RM = st.sidebar.slider('RM', min_value=float(X.RM.min()),max_value= float(X.RM.max()), step=float(X.RM.mean()))
    AGE = st.sidebar.slider('AGE', min_value=float(X.AGE.min()), max_value=float(X.AGE.max()),step=float( X.AGE.mean()))
    DIS = st.sidebar.slider('DIS', min_value=float(X.DIS.min()),max_value=float(X.DIS.max()), step=float(X.DIS.mean()))
    RAD = st.sidebar.slider('RAD', min_value=float(X.RAD.min()), max_value=float(X.RAD.max()), step=float(X.RAD.mean()))
    TAX = st.sidebar.slider('TAX',min_value=float(X.TAX.min()), max_value=float(X.TAX.max()),step=float( X.TAX.mean()))
    PTRATIO = st.sidebar.slider('PTRATIO',min_value=float(X.PTRATIO.min()),max_value=float(X.PTRATIO.max()), step=float(X.PTRATIO.mean()))
    B = st.sidebar.slider('B',min_value=float(X.B.min()),max_value=float(X.B.max()), step=float(X.B.mean()))
    LSTAT = st.sidebar.slider('LSTAT',min_value=float(X.LSTAT.min()),max_value=float(X.LSTAT.max()), step=float( X.LSTAT.mean()))
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')


# Build Regression Model

# Apply Model to Make Prediction
# prediction = model.predict([['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT']])
prediction = model.predict(df)

st.header('Prediction of House Price')
st.write(prediction)
st.write('---')
st.success('Done!... predicted Boston House price')


st.title('Dataset Information')
st.markdown("Boston House Prices Dataset was collected in 1978 and has 506 entries with 14 attributes or features for homes from various suburbs in Boston.")
st.markdown("""
    CRIM -  per capita crime rate by town                       
     ZN       proportion of residential land zoned for lots over 25,000 sq.ft.              
    INDUS    proportion of non-retail business acres per town           
    CHAS     Charles  River dummy variable (= 1 if tract bounds river; 0 otherwise)     
    NOX      nitric oxides concentration (parts per 10 million)             
     RM       average number of rooms per dwelling      
    AGE      proportion of owner-occupied units built prior to 1940      
    DIS      weighted distances to five Boston employment centres   
    RAD      index of accessibility to radial highways      
    TAX      full-value property-tax rate per $10,000  
""")
st.markdown("""
    PTRATIO  pupil-teacher ratio by town B 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town               
    LSTAT    % lower status of the population    Price     Median value of owner-occupied homes in $1000's 
""")



