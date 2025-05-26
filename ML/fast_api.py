from fastapi import FastAPI

from pydantic import BaseModel

import uvicorn

import pandas as pd 

import joblib

model = joblib.load('iris.pkl')

class LoanPred(BaseModel):

    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float 

app = FastAPI(title='Iris Prediction')

@app.get('/')
def get():
    return {'Message : Welcome to Iris prediction'}

@app.post('/predict')

def predict(loan_details : LoanPred):

    data = loan_details.model_dump()

    new_data = {
        'sepal_length': data['sepal_length'],
        'sepal_width': data['sepal_width'],
        'petal_length' : data['petal_length'], 
        'petal_width': data['petal_width']
    }


    df = pd.DataFrame([new_data])

    prediction = model.predict(df)

    if prediction[0]==0:
        pred = 'setosa'

    elif prediction[0]==1:
        pred = 'versicolor'

    else : 
        pred = 'virginica'
        
        
    return {"Model Prediction": pred}



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)








