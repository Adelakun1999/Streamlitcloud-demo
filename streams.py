import streamlit as st 
import pandas as pd 
import numpy as np 
import joblib
import warnings



model = joblib.load('iris.pkl')
st.title('Iris Flower Dataset')


#The input data 

data = {
    'sepal length (cm)' : st.number_input('Enter the sepal length (cm)'),
    'sepal width (cm)' : st.number_input('Enter the sepal width (cm)'),
    'petal length (cm)' : st.number_input('Enter the petal length (cm)'),
    'petal width (cm)' : st.slider('Enter the petal width (cm)')
}

df = pd.DataFrame([data])

if st.button('Show Data'):
    st.dataframe(df)


def prediction():
    pred = model.predict(df)
    if st.button('prediction'):
        return st.success(f'The predicted class is : {pred[0]}')
    

if __name__ == "__main__":
    prediction()