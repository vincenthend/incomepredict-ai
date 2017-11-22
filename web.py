from flask import Flask
from flask import render_template
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('form.html')
	
@app.route('/result')
def result():
	fitResult=[]
	
	
	return render_template('result.html',result=fitResult)