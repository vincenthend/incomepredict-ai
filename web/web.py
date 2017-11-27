# Flask Import 
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
app = Flask(__name__)

# Scikit & Pandas Import 
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
import pickle

@app.route('/')
def index():
    return render_template('form.html')
	
@app.route('/result', methods=['POST'])
def result():
	column = ['age', 'education-num', 'capital-gain', 'capital-loss',
       'hours-per-week', 'workclass_Federal-gov',
       'workclass_Local-gov', 'workclass_Never-worked', 'workclass_Private',
       'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc',
       'workclass_State-gov', 'workclass_Without-pay', 'education_10th',
       'education_11th', 'education_12th', 'education_1st-4th',
       'education_5th-6th', 'education_7th-8th', 'education_9th',
       'education_Assoc-acdm', 'education_Assoc-voc', 'education_Bachelors',
       'education_Doctorate', 'education_HS-grad', 'education_Masters',
       'education_Preschool', 'education_Prof-school',
       'education_Some-college', 'marital-status_Divorced',
       'marital-status_Married-AF-spouse', 'marital-status_Married-civ-spouse',
       'marital-status_Married-spouse-absent', 'marital-status_Never-married',
       'marital-status_Separated', 'marital-status_Widowed',
       'occupation_Adm-clerical', 'occupation_Armed-Forces',
       'occupation_Craft-repair', 'occupation_Exec-managerial',
       'occupation_Farming-fishing', 'occupation_Handlers-cleaners',
       'occupation_Machine-op-inspct', 'occupation_Other-service',
       'occupation_Priv-house-serv', 'occupation_Prof-specialty',
       'occupation_Protective-serv', 'occupation_Sales',
       'occupation_Tech-support', 'occupation_Transport-moving',
       'relationship_Husband', 'relationship_Not-in-family',
       'relationship_Other-relative', 'relationship_Own-child',
       'relationship_Unmarried', 'relationship_Wife', 'sex_Female',
       'sex_Male']
	
	
	
	request_data = {
		"age" : int(request.form['age']),
		"workclass" : request.form['workclass'],
		"fnlwgt" : int(request.form['fnlwgt']),
		"education" : request.form['education'],	
		"education-num" : int(request.form['education-num']),
		"marital-status" : request.form['marital-status'],
		"occupation" : request.form['occupation'],
		"relationship" : request.form['relationship'],
		"race" : request.form['race'],
		"sex" : request.form['sex'],
		"capital-gain" : int(request.form['capital-gain']),
		"capital-loss" : int(request.form['capital-loss']),
		"hours-per-week" : int(request.form['hours-per-week']),
		"native-country" : request.form['native-country']
	}
	
	dataFit = DataFrame(data=request_data, index=[0])
	dataFit = dataFit.drop(columns=['fnlwgt','race', 'native-country'])
	dataFit = pd.get_dummies(dataFit, columns=["workclass","education","marital-status","occupation","relationship","sex"])
	
	dataFrame = DataFrame(columns = column)
	dataFrame = pd.concat([dataFrame, dataFit], axis=0)
	dataFrame = dataFrame.fillna(0)
	
	fitResult = ""	
	tree_model = pickle.load(open("out.pkl","rb+"))
	fitResult = tree_model.predict(dataFrame)
	
	#return dataFrame.to_json(orient='index')
	return request
	#return jsonify(fitResult[0])