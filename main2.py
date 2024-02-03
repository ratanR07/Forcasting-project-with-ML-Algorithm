from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
from fastapi.encoders import jsonable_encoder
from sklearn.ensemble import BaggingClassifier


# Import xgboost
import xgboost as xgb

class InputData(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float
    OBV: float
    RSI: float
    Fib_1: float
    Fib_2: float
    Fib_3: float
    Fib_4: float
    K: float
    D: float
    SAR: float
    EP: float
    AF: float
    DI_plus: float
    DI_minus: float
    ADX: float
    MACD: float
    Signal_Line: float
    MACD_Histogram: float
    Middle_Band: float
    Upper_Band: float
    Lower_Band: float
    sin_high: float
    sin_low: float
    tan_high: float
    tan_low: float
    cos_high: float
    cos_low: float
    arctan_high: float
    arctan_low: float
    log_high: float
    log_low: float
    sqrt_high: float
    sqrt_low: float
    log_volume: float

# Load xgboost model
with open('Bagging_forex_model.pkl', 'rb') as f:
    model1 = pickle.load(f)

app = FastAPI()

@app.post('/pred')
def get_prediction(data: InputData):
    received = pd.DataFrame(jsonable_encoder(data), index=[0])
    cols_new = ['open', 'high', 'low', 'close', 'volume', 'OBV',
                'RSI', 'Fib_1', 'Fib_2', 'Fib_3', 'Fib_4', 'K', 'D', 'SAR', 'EP', 'AF',
                'DI_plus', 'DI_minus', 'ADX', 'MACD', 'Signal_Line', 'MACD_Histogram',
                'Middle_Band', 'Upper_Band', 'Lower_Band', 'sin_high', 'sin_low',
                'tan_high', 'tan_low', 'cos_high', 'cos_low', 'arctan_high',
                'arctan_low', 'log_high', 'log_low', 'sqrt_high', 'sqrt_low',
                'log_volume']

    received = received[cols_new]
    pred_name = model1.predict(received)[0]
    Prob = model1.predict_proba(received) * 100
    probability_percent = {
        "Fraud_odds": round(Prob.tolist()[0][0], 2),
        "NoFraud_odds": round(Prob.tolist()[0][1], 2)
    }
    decimal_odds = {
        "Fraud_odds": round(100/Prob.tolist()[0][0], 2),
        "NoFraud_odds": round(100/Prob.tolist()[0][1], 2)
    }

    return {'prediction': pred_name,
            'probability_percent': probability_percent,
            "predicted_decimal_odds": decimal_odds
            }
