import sklearn
import joblib
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBClassifier
from flask import Flask, request
import imblearn

pipeline = joblib.load('finalized_model.sav')

print(pipeline)

# Demarrer l'API
app = Flask('__name__')

@app.route('/predict', methods=['POST'])
def predict():
  df = pd.DataFrame(request.json)

  resultat = pipeline.predict(df)[0]
  
  return (str(resultat), 201)




@app.route('/ping', methods=['GET'])
def ping():
  return('pong', 200)




# Definir une page d'accueil
@app.route('/')
def index():
  return "<h1>Bienvenue dans ntre API. Utiliser /predict en POST pour faire des prédictions.</h1>"

# Si on est dans le "main", on lance l'API
if __name__=="__main__":
  app.run(host='0.0.0.0')







