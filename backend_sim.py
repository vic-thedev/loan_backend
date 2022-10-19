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

placebo = pd.read_csv('default_value_ultimate.csv')
placebo = np.array(placebo).reshape(1, -1)


@serve.deployment(route_prefix="/loan")
@serve.ingress(app)
class BoostingModel:
    def __init__(self, model):
        self.model = model

    @app.get("/")
    async def root(self, request: Request):
        payload = await request.json()
        print(payload)
        prediction = self.model.predict(np.array(payload["vector"]).reshape(1, -1))[0]
        human_name = {0: "refused", 1: "accepted"}
        return {"result": human_name[prediction]}

    @app.post("/sub")
    async def root(self, that : Request):
        dt = await that.json()
        ligne = f"Hello, {dt['name']}, how are you?"
        return {"id": ligne}


if __name__ == "__main__":
    # Chargement du modele
    model = joblib.load('ultimate_model.modele')
    placebo_result = model.predict(placebo)

    # Données génériques
    sample_request_input = {"vector": placebo.tolist()}
    req_imp_json = json.dumps(sample_request_input)
    serve.run(BoostingModel.bind(model), port=80)
    #exemple de plus simplke
    '''from fastapi import FastAPI, Depends
    from ray.serve.http_adapters import json_to_ndarray

    app = FastAPI()

    @app.post("/endpoint")
    async def endpoint(np_array = Depends(json_to_ndarray)):
        ...'''
