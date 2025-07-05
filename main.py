from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI(title="CHD MCP Server", description="MCP-style endpoint for cardiovascular risk prediction")

model = pickle.load(open("model_lr_ch1.pkl", "rb"))

class CHDInput(BaseModel):
    input_male: int
    age: int
    education: int
    currentSmoker: int
    cigsPerDay: float
    BPMeds: int
    prevalentStroke: int
    prevalentHyp: int
    diabetes: int
    totChol: float
    sysBP: float
    diaBP: float
    BMI: float
    heartRate: float
    glucose: float

class CHDOutput(BaseModel):
    prediction: int
    risk: str

@app.get("/")
def read_root():
    return {"message": "CHD MCP Server is running."}

@app.post("/predict", response_model=CHDOutput)
def predict_chd(input_data: CHDInput):
    data = [[
        input_data.input_male, input_data.age, input_data.education,
        input_data.currentSmoker, input_data.cigsPerDay, input_data.BPMeds,
        input_data.prevalentStroke, input_data.prevalentHyp, input_data.diabetes,
        input_data.totChol, input_data.sysBP, input_data.diaBP, input_data.BMI,
        input_data.heartRate, input_data.glucose
    ]]

    prediction = model.predict(np.array(data))[0]
    risk = "high" if prediction == 1 else "low"
    return CHDOutput(prediction=int(prediction), risk=risk)
