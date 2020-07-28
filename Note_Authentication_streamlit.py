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
#import flasgger
#from flasgger import Swagger
import streamlit as st
from PIL import Image

#Step1: always start with this line and what is __name__
#The defination is below
#app = Flask(__name__)
#Swagger(app)

#step 3: load pickle file into application
pickle_in = open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)


#Step 4: what will be the route or when ip hit which page should return
#by defualte rout take get method
#@app.route('/')
def welcome():
    return "welcome All"

#Step 5: create function to predict
#@app.route('/predict', methods=["Get"])
def predict_note_authentication(variance, skewness,curtosis, entropy):
    #please make sure you follow indentation
    """Let's Authenticate the Banks Note
    This is using docstring for specifications.
    ---
    parameters:
        -   name: variance
            in: query
            required: true
        -   name: skewness
            in: query
            type: number
            required: true
        -   name: curtosis
            in: query
            type: number
            required: true
        -   name: entropy
            in: query
            type: number
            required: true
    responses:
        200:
            description: The output values
        
    """

    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return prediction

#now we use file data to test the result
#@app.route('/predict_file', methods=["POST"])
def predict_note_file():
    """Let's Authenticate the Banks Note
    This is using docstrings for specificaitons.
    ---
    parameters:
        - name: file
          in: formData
          type: file
          required: true
    responses:
        200:
            description: The output values
    """
    df_test = pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(df_test)
    return "The predicted value for csv file is "+ str(list(prediction))

def main():
    st.title("Bank Authenticator")
    html_temp = """
    <div style = "background-color:tomato; padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Note Authenticator ML App</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    variance = st.text_input("variance","Type Here")
    skewness = st.text_input("skewness","Type Here")
    curtosis = st.text_input("curtosis","Type Here")
    entropy = st.text_input("entropy","Type Here")
    result = ""
    if st.button("Predict"):
        result = predict_note_authentication(variance, skewness,curtosis, entropy)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Nuilt with Steamlit")
    
    
    
    
    
#Step 2: address the name function
if __name__ == '__main__':
    main()
   # app.run()
    
    
    
