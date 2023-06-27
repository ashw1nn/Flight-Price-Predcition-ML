from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            airline=request.form.get('airline'),
            source=request.form.get('source'),
            destination=request.form.get('destination'),
            additional_info=request.form.get('additional_info'),
            duration=float(request.form.get('duration')),
            total_stops=float(request.form.get('total_stops')),
            date=float(request.form.get('date')),
            month=float(request.form.get('month')),
            year=float(request.form.get('year')),
            arrival_hour=float(request.form.get('arrival_hour')),
            arrival_minute=float(request.form.get('arrival_minute')),
            dep_hour=float(request.form.get('dep_hour')),
            dep_minute=float(request.form.get('dep_minute')),
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)        


