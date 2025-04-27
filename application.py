import joblib
from flask import Flask,render_template,request
import numpy as np
import pandas as pd

app=Flask(__name__)

MODEL_PATH="artifacts/models/model.pkl"
model=joblib.load(MODEL_PATH)

FEATURES= ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 
           'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 
           'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 
           'RainToday', 'Year', 'Month', 'Day']

LABELS={0:"NO",1:"YES"}

@app.route("/",methods=["GET","POST"])
def index():
    prediction=None

    if request.method=="POST":
        try:
            input_data=[float(request.form[feature]) for feature in FEATURES]
            print(f"Received input data: {input_data}")

            input_array=np.array(input_data).reshape(1,-1)
            print(f"Reshaped input array: {input_array}")

            input_df = pd.DataFrame(input_array, columns=FEATURES)
            print(f"Data as DataFrame: {input_df}")

            pred = model.predict(input_df)[0]
            prediction=LABELS.get(pred,'Unknown')
            print(f"Prediction: {prediction}")
        except Exception as e:
            print(f"Error occurred: {str(e)}")

    return render_template("index.html",prediction=prediction,features=FEATURES)

if __name__=="__main__":
    app.run(debug=True, port=5000,host="0.0.0.0")

