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


@app.route('/predict_proba', methods=['POST'])
def predict_proba():
  df = pd.DataFrame(request.json)

  resultat = pipeline.predict_proba(df)
  resultat = pd.DataFrame(resultat)
  
  return (resultat.to_json(), 201)

 

seuil = 0.316033

@app.route('/predict_proba_seuil', methods=['POST'])
def predict_proba_seuil():
  df = pd.DataFrame(request.json)

  resultat = (pipeline.predict_proba(df)[:,1]>= seuil).astype(bool)
  
  return (str(resultat), 201)


@app.route('/ping', methods=['GET'])
def ping():
  return('pong', 200)



# Definir une page d'accueil
@app.route('/')
def index():
  return "<h1>Bienvenue dans notre API. Utiliser /predict et /predict_proba en POST pour faire des prédictions et des probabilités.</h1>"

# Si on est dans le "main", on lance l'API
if __name__=="__main__":
  app.run(host='0.0.0.0')

