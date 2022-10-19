import numpy as np
import ray
import requests
import pandas as pd
import json
import joblib
from fastapi import FastAPI, Request , Depends
from ray.serve.http_adapters import json_to_ndarray
from ray import serve

app = FastAPI()

#if __name__ == "__main__":
# Chargement du modele
model = joblib.load('ultimate_model.modele')
placebo = pd.read_csv('default_value_ultimate.csv')
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
    payload = await features.json()
    #payload =  Depends(json_to_ndarray)
    print(payload)
    prediction = model.predict(np.array(payload["vector"]).reshape(1, -1))[0]
    human_name = {0: "refused", 1: "accepted"}
    return {"result": human_name[prediction]}

@app.post("/sub")
async def root(that : Request):
    dt = await that.json()
    ligne = f"Hello, {dt['name']}, how are you?"
    return {"id": ligne}


