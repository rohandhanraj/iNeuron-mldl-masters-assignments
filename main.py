"""
author: Rohan Dhanraj
email: rdy5674@gmail.com
""" 

import numpy as np
import pandas as pd
from utils import * # preprocessing and transforming data....

from wsgiref import simple_server
from flask import Flask, render_template, request, Response
from flask_cors import CORS,cross_origin

from xgboost import XGBClassifier

import os
import pickle
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename = os.path.join(log_dir,'running_logs.log'), level=logging.INFO, format=logging_str)



os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET', 'POST'])  # To render Homepage
@cross_origin()
def home_page():
    return render_template('index.html')

@app.route('/linearRegression', methods=['GET', 'POST'])  # To render Homepage
@cross_origin()
def lin_reg():
    return render_template('boston.html')

@app.route('/logisticRegression', methods=['GET', 'POST'])  # To render Homepage
@cross_origin()
def log_reg():
    return render_template('affairs.html')

@app.route('/decisionTree', methods=['GET', 'POST'])  # To render Homepage
@cross_origin()
def dt_clf():
    return render_template('titanic.html')

@app.route('/randomForest', methods=['GET', 'POST'])  # To render Homepage
@cross_origin()
def rf_reg():
    return render_template('boston_rf.html')

@app.route('/xgBoost', methods=['GET', 'POST'])  # To render Homepage
@cross_origin()
def xgb_clf():
    return render_template('adult_income.html')


@app.route('/xgBoost-result', methods=['POST'])  # To render Result page
@cross_origin()
def xgb_result():
    if request.method == 'POST':
        try:
            input_dict = request.form.to_dict()
            logging.info(input_dict)
            input_features = {k.strip(): [try_typecasting(float, v) ] for k, v in input_dict.items()}
            logging.info(input_features)
            
            data = pd.DataFrame(input_features)
            logging.info(data)

            model = pickle.load(open('models/xgBoostPipeline.sav', 'rb'))
            pred = model.predict(data)[0]
            result = f'The income is {"LESS THAN $50K" if pred == 0 else "MORE THAN $50K"}.'
            logging.info(f'Result:\n{"-----"*10} \n{result}')
            logging.info("====="*10)

            return render_template('result.html', result=result)
            
        except Exception as e:
            logging.info('The Exception message is: \n',e)
            return 'Something went wrong'+str(e)
    else:
        return render_template('adult.html')


@app.route('/randomForest-result', methods=['POST'])  # To render Result page
@cross_origin()
def rf_result():
    if request.method == 'POST':
        try:
            input_dict = request.form.to_dict()
            logging.info(input_dict)
            input_features = {k.strip(): [float(v)] for k, v in input_dict.items()}
            logging.info(input_features)
            
            data = pd.DataFrame(input_features)

            model = pickle.load(open('models/randomForestPipeline.sav', 'rb'))

            pred = model.predict(data)[0]
            result = f'The price estimate of house is ${pred * 1000}.'
            logging.info(f'Result:\n{"-----"*10} \n{result}')
            logging.info("====="*10)

            return render_template('result.html', result=result)
            
        except Exception as e:
            logging.info('The Exception message is: \n',e)
            return 'Something went wrong'
    else:
        return render_template('boston_rf.html')


@app.route('/decisionTree-result', methods=['POST'])  # To render Result page
@cross_origin()
def dt_result():
    if request.method == 'POST':
        try:
            input_dict = request.form.to_dict()
            logging.info(input_dict)
            input_features = {k.strip(): [v] for k, v in input_dict.items()}
            logging.info(input_features)
            data = pd.DataFrame(input_features)
            logging.info('DataFrame Created:\n', data)

            model = pickle.load(open('models/decisionTreeClassifier.sav', 'rb'))

            data = titanic_preprocess(data)
            logging.info('Preprocessed Data:\n', data)
            data = titanic_transformer(data)
            logging.info('Transformed Data:\n', data)
            data = titanicDropFeatures(data)
            logging.info('After drop:\n', data)

            pred = model.predict(data)[0]
            logging.info('Prediction:', pred)
            result = f'The passenger {"Survived" if pred == 1 else "Died"}.'
            logging.info(f'Result:\n{"-----"*10} \n{result}')
            logging.info("====="*10)

            return render_template('result.html', result=result)
            
            
        except Exception as e:
            logging.info('The Exception message is: \n',e)
            return 'Something went wrong'
    else:
        return render_template('titanic.html')


@app.route('/logisticRegression-result', methods=['POST'])  # To render Result page
@cross_origin()
def log_result():
    if request.method == 'POST':
        try:
            input_dict = request.form.to_dict()
            logging.info(input_dict)
            input_features = {k.strip(): [float(v)] for k, v in input_dict.items()}
            logging.info(input_features)
            data = pd.DataFrame(input_features)

            cols, model = pickle.load(open('models/logisticRegressionClassifier.sav', 'rb'))

            #data = pd.DataFrame({'rate_marriage': [3.0], 'age': [27.0], 'yrs_married': [3.0], 'children': [2.0], 'religious': [3.0], 'educ': [16.0], 'occupation': [3.0], 'occupation_husb': [5.0]})
            data[cols[0]] = 1.0
            data[cols[1:11]] = 0

            for i, j in zip(cols[1:6], cols[6:11]) :
                if float(i[-1]) == data['occupation'].values[0]:
                    data[i] = 1
                elif float(j[-1]) == data['occupation_husb'] .values[0]:
                    data[j] = 1
            else:
                x = data[cols]

            pred = model.predict(x)[0]
            predictions = ("Yes", " ") if pred == 1 else ("No", 'do not ')
            result = f'{predictions[0].capitalize()} the woman {predictions[1].capitalize()}have at least one affair.'
            logging.info(f'Result:\n{"-----"*10} \n{result}')
            logging.info("====="*10)

            return render_template('result.html', result=result)
            
        except Exception as e:
            logging.info('The Exception message is: \n',e)
            return 'Something went wrong'
    else:
        return render_template('affairs.html')


@app.route('/linearRegression-result', methods=['POST'])  # To render Result page
@cross_origin()
def lr_result():
    if request.method == 'POST':
        try:
            input_dict = request.form.to_dict()
            logging.info(input_dict)
            input_features = {k.strip(): [float(v)] for k, v in input_dict.items()}
            logging.info(input_features)
            data = pd.DataFrame(input_features)

            model = pickle.load(open('models/linearRegressorPipeline.sav', 'rb'))

            pred = model.predict(data)[0]
            result = f'The price estimate of house is ${pred * 1000}.'
            logging.info(f'Result:\n{"-----"*10} \n{result}')
            logging.info("====="*10)

            return render_template('result.html', result=result)
            
        except Exception as e:
            logging.info('The Exception message is: \n',e)
            return 'Something went wrong'
    else:
        return render_template('boston.html')

port = int(os.getenv("PORT", 5000))
if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000, debug=True)
    host = '0.0.0.0'
    # port = 5000
    httpd = simple_server.make_server(host, port, app)
    logging.info("Serving on %s %d" % (host, port))
    httpd.serve_forever()