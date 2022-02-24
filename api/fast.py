import uuid
from datetime import datetime

import joblib
import pandas as pd
import pytz
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}


@app.get("/predict")
def predict(pickup_datetime,
    pickup_longitude: float,
    pickup_latitude: float,
    dropoff_longitude: float,
    dropoff_latitude: float,
    passenger_count: int):

    pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
    eastern = pytz.timezone("US/Eastern")
    localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)
    utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)
    formatted_pickup_datetime = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")


    key = datetime.now(tz=eastern).astimezone(pytz.utc)
    key=key.strftime("%Y%m%d%H%M%S%f")

    X_pred = pd.DataFrame([[key,
                          formatted_pickup_datetime,
                          pickup_longitude,
                          pickup_latitude,
                          dropoff_longitude,
                          dropoff_latitude,
                          passenger_count]],
                        columns=
    ["key",
     "pickup_datetime",
     "pickup_longitude",
     "pickup_latitude",
     "dropoff_longitude",
     "dropoff_latitude",
     "passenger_count"],
                         # dtype=[
                         #     "object",
                         #     "object",
                         #     "float64",
                         #     "float64",
                         #     "float64",
                         #     "float64",
                         #     "integer64",
                         # ]
                         )
    model = joblib.load("model.joblib")
    y_pred = model.predict(X_pred)
    datetime.strptime()
    return {"fare":y_pred[0]}

