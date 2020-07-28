#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 13:08:01 2020

@author: ganesh_dhasade
"""

#Import required libraries
from flask import Flask, request
import pickle
import pandas as pd
import numpy as np

#Step1: always start with this line and what is __name__
#The defination is below
app = Flask(__name__)


#step 3: load pickle file into application
pickle_in = open('classifier.pkl','rb')

classifier = pickle.load(pickle_in)
#Step 4: what will be the route or when ip hit which page should return
#by defualte rout take get method
@app.route('/')
def welcome():
    return "welcome All"

#Step 5: create function to predict
@app.route('/predict')
def predict_note_authentication():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    return "The predicted value is "+ str(prediction)

#now we use file data to test the result
@app.route('/predict_file', methods=["POST"])
def predict_note_file():
    df_test = pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(df_test)
    return "The predicted value for csv file is "+ str(list(prediction))

#Step 2: address the name function
if __name__ == '__main__':
    app.run()
    
    
    
