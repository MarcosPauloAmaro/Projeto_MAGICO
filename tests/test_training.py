import os
import subprocess
import sys
import joblib
import pandas as pd

def test_treino_gera_modelos():
    subprocess.run([sys.executable, "src/TRAIN_READ.py"], check=True)
    assert os.path.exists("models/model_class.pkl")
    assert os.path.exists("models/model_reg.pkl")

def test_modelos_predizem():
    clf = joblib.load("models/model_class.pkl")
    reg = joblib.load("models/model_reg.pkl")

    df = pd.DataFrame([{
        "Ano nasc": 2007,
        "Idade 22": 15,
        "Ano ingresso": 2021,
        "IAA": 7.5,
        "IEG": 6.0,
        "IPS": 7.5,
        "IDA": 2.7,
        "IPV": 4.7,
        "IAN": 5.0,
        "Matem": 6.0,
        "Portug": 6.0,
        "Ingles": 6.0,
        "Gênero": "Menino",
        "Fase": "3",
        "Turma": "B",
    }])

    pred_class = clf.predict(df)[0]
    pred_reg = reg.predict(df)[0]

    assert pred_class in [0, 1]
    assert isinstance(float(pred_reg), float)