import numpy as np
import ray
import requests
import pandas as pd
import json
import joblib
from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse
import dill as pickle
import lime
import lime.lime_tabular
from ray.serve.http_adapters import json_to_ndarray
from ray import serve


app = FastAPI()

#############Initialisation explicabilité avec lime################
#Explainer initialisation
explainer = pickle.load(open('explainer777.pkl', 'rb'))
most_imp = ['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_REGISTRATION',
'DAYS_ID_PUBLISH', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
'LIVINGAREA_AVG', 'DAYS_LAST_PHONE_CHANGE', 'INCOME_CREDIT_PERC',
'INCOME_PER_PERSON', 'ANNUITY_INCOME_PERC', 'PAYMENT_RATE',
'DAYS_EMPLOYED_PERC', 'APPS_EXT_SOURCE_MEAN', 'APPS_EXT_SOURCE_STD',
'APP_EXT_SOURCE_2*EXT_SOURCE_3*DAYS_BIRTH', 'CREDIT_TO_GOODS_RATIO',
'INCOME_TO_EMPLOYED_RATIO', 'INCOME_TO_BIRTH_RATIO',
'ID_TO_BIRTH_RATIO', 'CAR_TO_BIRTH_RATIO']


#if __name__ == "__main__":
# Chargement du modele
model = joblib.load('ultimate_model_f777.modele')
placebo = pd.read_csv('default_value777.csv')
placebo = np.array(placebo).reshape(1, -1)
placebo_result = model.predict(placebo)

# Données génériques
sample_request_input = {"vector": placebo.tolist()}
req_imp_json = json.dumps(sample_request_input)
#serve.run(BoostingModel.bind(model), port=8080)

@app.get("/")
async def f():
    return {"message": "Bienvenue sur la meilleure application de prédiction de capacité à rembourser des prêts bancaires"}

#@serve.deployment(route_prefix="/")
#@serve.ingress(app)
class BoostingModel:
    def __init__(self, model):
        self.model = model



@app.get("/loan")
async def root(features: Request):
    print('load request received')
    payload = await features.json()
    #payload =  Depends(json_to_ndarray)
    print(payload)
    prediction = model.predict(np.array(payload["vector"]).reshape(1, -1))[0]
    pb = model.predict_proba(np.array(payload["vector"]).reshape(1, -1))[0][1]
    print(pb)
    human_name = {0: "Pas de difficulté particulière à rembourser le prêt", 1: "Fort risque de nom remboursement du prêt"}
    return {"result": f"{human_name[prediction]}. Voici votre score : {pb * 100}"}

@app.post("/lime")
async def root(features: Request):
    print("Request received")
    payload = await features.json()
    print(payload)
    borrower = pd.Series(payload["newclient"])
    print(borrower)
    predict_fn = lambda x: model.predict_proba(x).astype(float)
    exp = explainer.explain_instance(borrower, predict_fn, num_features=20)
    exp_html= exp.as_html()
    print()
    return HTMLResponse(exp_html)



@app.post("/sub")
async def root(that : Request):
    dt = await that.json()
    ligne = f"Hello, {dt['name']}, how are you?"
    return {"id": ligne}


