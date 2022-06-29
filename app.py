
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import pandas as pd

from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

Final_model= pickle.load(open('creditcard.pkl', 'rb'))


@app.route('/', methods=['GET'])

def Home():
    return render_template('index1.html')


@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST':
        years = int(request.form['years'])
        amount=float(request.form['amount'])
        grade=request.form['grade']
        ownership=request.form['ownership']
        income=int(request.form['income'])
        age=int(request.form['age'])

        SampleInputData=pd.DataFrame(
            data=[[amount, grade, years, ownership,income, age]],
            columns=['amount', 'grade','years','ownership','income', 'age'])
        Num_Inputs=SampleInputData.shape[0]
        DataForML= pickle.load(open('DataForML.pkl', 'rb'))
        SampleInputData=SampleInputData.append(DataForML)
        SampleInputData=pd.get_dummies(SampleInputData)
        X=SampleInputData.values[0:Num_Inputs]

        with open('creditcard.pkl', 'rb') as fileReadStream:
                Final_model=pickle.load(fileReadStream)
                fileReadStream.close()
                Prediction=Final_model.predict(X)

        if Prediction == 0:
            return render_template('Result.html',prediction_texts="Sorry You are not eligible for credit card")
        else:
            return render_template('Result.html',prediction_texts="Congrats!!! You are eligible for credit card")  
       
    else:   
        return render_template('index1.html')


if __name__=="__main__":
    app.run(debug=True)