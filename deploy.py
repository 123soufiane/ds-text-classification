from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model/text_model.joblib")

class TextIn(BaseModel):
    text: str

@app.post("/predict")
def predict(inp: TextIn):
    return {"prediction": model.predict([inp.text])[0]}
