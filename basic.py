import numpy as np
import ray
import requests
import pandas as pd
import json
import joblib
from fastapi import FastAPI, Request, Depends
from ray.serve.http_adapters import json_to_ndarray
from ray import serve

app = FastAPI()

placebo = pd.read_csv('default_value_ultimate.csv')
placebo = np.array(placebo).reshape(1, -1)


@app.get("/")
async def root():
    return {"message": "Hello World"}


