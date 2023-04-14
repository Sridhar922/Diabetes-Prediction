#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import pickle


# In[2]:


#create flask app
app= Flask(__name__)


# In[3]:


scaler=pickle.load(open('scaler.pkl','rb'))
model=pickle.load(open('diabetes.pkl','rb'))


# In[4]:


## Route for homepage
@app.route('/')
def index():
    return render_template('home.html')


# In[5]:


## Route for Single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=""

    if request.method=='POST':

        Pregnancies=int(request.form.get("Pregnancies"))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        new_data=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict=model.predict(new_data)
       
        if predict[0] ==1 :
            result = 'Diabetic'
        else:
            result ='Non-Diabetic'
            
        return render_template('single_prediction.html',result=result)

    else:
        return render_template('home.html')


# In[ ]:


if __name__=="__main__":
    app.run(debug=True,use_reloader=False)


# In[ ]:




