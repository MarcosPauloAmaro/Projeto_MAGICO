# src/API.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# 🔎 MONITORAMENTO / LOGS
import json
from datetime import datetime
import os

print("Carregando modelos...")

clf = joblib.load("models/model_class.pkl")
reg = joblib.load("models/model_reg.pkl")

LOG_PATH = "logs/predict_logs.jsonl"
os.makedirs("logs", exist_ok=True)

app = FastAPI(title="API Passos Mágicos - Defasagem Escolar")

# ==============================
# SCHEMA INPUT (campos que o Streamlit envia)
# ==============================
class Aluno(BaseModel):
    Ano_nasc: float
    Idade_22: float
    Ano_ingresso: int
    Genero: str
    Fase: str
    Turma: str
    IAA: float
    IEG: float
    IPS: float
    IDA: float
    IPV: float
    IAN: float
    Matem: float
    Portug: float
    Ingles: float

# ==============================
# ENDPOINT
# ==============================
@app.post("/predict")
def predict(aluno: Aluno):
    try:
        dados = aluno.dict()

        # Mapear exatamente para os nomes do Excel usados no treino
        df = pd.DataFrame([{
            "Ano nasc": dados["Ano_nasc"],
            "Idade 22": dados["Idade_22"],
            "Ano ingresso": dados["Ano_ingresso"],
            "IAA": dados["IAA"],
            "IEG": dados["IEG"],
            "IPS": dados["IPS"],
            "IDA": dados["IDA"],
            "IPV": dados["IPV"],
            "IAN": dados["IAN"],
            "Matem": dados["Matem"],
            "Portug": dados["Portug"],
            "Ingles": dados["Ingles"],
            "Gênero": dados["Genero"],
            "Fase": dados["Fase"],
            "Turma": dados["Turma"],
        }])

        pred_class = int(clf.predict(df)[0])
        prob = float(clf.predict_proba(df)[0][1])
        pred_reg = float(reg.predict(df)[0])

        # 📌 LOG DE MONITORAMENTO
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": datetime.utcnow().isoformat(),
                "input": dados,
                "output": {
                    "risco": pred_class,
                    "prob": prob,
                    "defasagem": pred_reg
                }
            }) + "\n")

        return {
            "risco_defasagem": pred_class,
            "probabilidade_risco": round(prob, 4),
            "defasagem_prevista": round(pred_reg, 2)
        }

    except Exception as e:
        print("ERRO NO /predict:", e)
        raise HTTPException(status_code=500, detail=str(e))

# uvicorn src.API:app --reload