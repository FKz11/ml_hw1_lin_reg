import pickle
import pandas as pd

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from preproc import preproc

with open("models/lin_reg.pkl", "rb") as f:
    lin_reg = pickle.load(f)

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df_numpy = preproc(pd.DataFrame([dict(item)]))
    selling_price = lin_reg.predict(df_numpy)
    return max(selling_price, 0.)


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    df_numpy = preproc(pd.DataFrame([dict(item) for item in items]))
    selling_prices = lin_reg.predict(df_numpy)
    return [max(selling_price, 0.) for selling_price in selling_prices]


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
