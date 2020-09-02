# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from flask import Flask, render_template, request,redirect
import flask_monitoringdashboard as dashboard
from flask import send_file
from flask import Response
from wsgiref import simple_server
import os
from flask_cors import CORS, cross_origin
import flask_monitoringdashboard as dashboard
from upload_file.uploadFile import UploadFile
from data_preprocessing.dataPreprocessing import Preprocessor
#from data_validation.dataValidation import DataValidation
from prediction.predictFromModel import Prediction
from training.trainingModel import Training

import csv
import jsonify
import requests
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
dashboard.bind(app)
app.secret_key = "secret key"
model = pickle.load(open('SVC_rbf_model_v1.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('start.html')

@app.route('/',methods = ['POST'])
def start():
    if request.method == 'POST':
        input_type = request.form['input_type']
        if input_type=='single':
            return render_template('index.html')
        elif input_type == 'file':
            return render_template('index_1.html')        
        else:
            return redirect('/')

@app.route("/predict-api", methods=['POST'])
def SingleRecordPrediction():    
    if request.method == 'POST':
        Price=request.form['Price']
        if(Price=='low'):
            Price=0                
        elif(Price=='med'):
            Price=1
        elif(Price=='high'):
            Price=2
        else:
            Price=3
        MaintenanceCost=request.form['Maintenance Cost']
        if(MaintenanceCost=='low'):
            MaintenanceCost=0                
        elif(MaintenanceCost=='med'):
            MaintenanceCost=1
        elif(MaintenanceCost=='high'):
            MaintenanceCost=2
        else:
            MaintenanceCost=3
        NumberOfDoors=request.form['Number of Doors']
        if(NumberOfDoors=='5more'):
            NumberOfDoors=5  
        Capacity=request.form['Capacity']
        if(Capacity=='more'):
            Capacity=5            
        SizeOfLuggageBoot=request.form['Size of Luggage Boot']
        if(SizeOfLuggageBoot=='small'):
            SizeOfLuggageBoot=0      
        elif(SizeOfLuggageBoot=='med'):
            SizeOfLuggageBoot=1
        else:
            SizeOfLuggageBoot=2 
        safety=request.form['safety']
        if(safety=='low'):
            safety=0                
        elif(safety=='med'):
            safety=1
        else:
            safety=2
            
        #standard_to = StandardScaler()
            
        prediction=model.predict([[Price,MaintenanceCost,NumberOfDoors,Capacity,SizeOfLuggageBoot,safety]])
        output=prediction
        if output<0:
            return render_template('results.html',prediction_texts="Sorry prediction has some problem")
        elif output==0:
            return render_template('results.html',prediction_text= "unaccounted")
        elif output==1:
            return render_template('results.html',prediction_text= "accounted")
        elif output==2:
            return render_template('results.html',prediction_text= "good")
        elif output==3:
            return render_template('results.html',prediction_text= "vgood")
    else:
        return render_template('index.html')
            
@app.route("/upload", methods=['POST'])
@cross_origin()
def upload():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return Response('No file selected for uploading')

            file = request.files['file']

            # Upload file Object Initialisation
            upld_file = UploadFile(file)
            # Calling the Upload file function
            resp_msg_upld = upld_file.upload_file(file)
            return Response(resp_msg_upld)

        except Exception as e:
            return Response("Error occured in uploading the file.", e)

    else:
        return render_template('index_1.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRouteClient():

    if request.method == 'POST':
        try:
            filepath_val = "Input_Files/Uploaded_file.csv"
            process_type = "P"
            # Pre-Processor Object Initialisation
            pred_prep = Preprocessor(filepath_val,process_type)
            # Calling the Data Pre-Processing function
            resp_msg_prep = pred_prep.data_preprocess(filepath_val,process_type)

            if resp_msg_prep == "Pre-Processing Success":
                # Prediction Object Initialisation
                pred = Prediction()
                # Predicting for the dataset uploaded by the user
                resp_msg_pred = pred.predict_model()
                return resp_msg_pred
            else:
                #return json.dumps(resp_msg_prep)
                return resp_msg_prep

        except Exception as e:
            return Response("Error occured during prediction", e)

    else:
        return render_template('index_1.html')
@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():
    if request.method == 'POST':

        try:
            filepath_val = "Input_Files/Uploaded_file.csv"
            process_type = "T"
            # Pre-Processor Object Initialisation
            train_prep = Preprocessor(filepath_val,process_type)
            # Calling the data_preprocess function
            resp_msg_train_prep =train_prep.data_preprocess(filepath_val,process_type)
            if resp_msg_train_prep == "Pre-Processing Success":
           
                # Training Object Initialisation
                train = Training()
                # Training for the dataset uploaded by the user
                resp_msg_train = train.train_model()
                return resp_msg_train
            else:
                return resp_msg_train_prep


        except Exception as e:
            return Response("Error occured during re-training", e)
    else:
        return render_template('index_1.html')
@app.route('/downloads/')
@cross_origin()
def file_downloads():
        try:
            return send_file('Predicted_Files/Result.csv', attachment_filename='Result.csv',as_attachment=True, cache_timeout=0)
        except Exception as e:
            return Response("Error occured during downloading result", e)

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r  




if __name__=="__main__":
    app.run(debug=True)

