# src/TRAIN_READ.py
import pandas as pd
import numpy as np
import os
import joblib
import sys
sys.stdout.reconfigure(encoding="utf-8")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

print("🚀 INICIANDO PIPELINE COMPLETO (CLASSIFICAÇÃO + REGRESSÃO)")

# ==============================
# 1️⃣ CARREGAR DADOS
# ==============================
CAMINHO_DADOS = "data/DADOS.xlsx"  # ajuste se necessário

print("\n📥 Lendo TODAS as abas do Excel...")
abas = pd.read_excel(CAMINHO_DADOS, sheet_name=None)
df = pd.concat(abas.values(), ignore_index=True)

print("\n===== DADOS CARREGADOS =====")
print(df.info())

# ==============================
# 2️⃣ FEATURES E TARGETS (NOMES REAIS DO EXCEL)
# ==============================
features_num = [
    "Ano nasc", "Idade 22", "Ano ingresso",
    "IAA", "IEG", "IPS", "IDA", "IPV", "IAN",
    "Matem", "Portug", "Ingles"
]

features_cat = ["Gênero", "Fase", "Turma"]

target_reg = "Defasagem"  # contínuo: -1 até 5

colunas_necessarias = features_num + features_cat + [target_reg]
faltando = [c for c in colunas_necessarias if c not in df.columns]
if faltando:
    raise ValueError(f"❌ Colunas faltando no Excel: {faltando}")

df = df[colunas_necessarias]

# ==============================
# 3️⃣ LIMPEZA
# ==============================
# target
df[target_reg] = pd.to_numeric(df[target_reg], errors="coerce")
df = df.dropna(subset=[target_reg])

# numéricas
for col in features_num:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

# categóricas sempre string
for col in features_cat:
    df[col] = df[col].astype(str).fillna("Desconhecido")

# classificação derivada: risco = defasagem > 0
df["Risco"] = (df[target_reg] > 0).astype(int)

print("✅ Dados tratados")
print("📊 Distribuição de Risco:")
print(df["Risco"].value_counts())

# ==============================
# 4️⃣ X e y
# ==============================
X = df[features_num + features_cat]
y_class = df["Risco"]
y_reg = df[target_reg]

# ==============================
# 5️⃣ PIPELINES
# ==============================
preprocessador = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), features_num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), features_cat),
    ]
)

clf = Pipeline(steps=[
    ("prep", preprocessador),
    ("model", RandomForestClassifier(n_estimators=300, random_state=42))
])

reg = Pipeline(steps=[
    ("prep", preprocessador),
    ("model", RandomForestRegressor(n_estimators=300, random_state=42))
])

# ==============================
# 6️⃣ SPLIT
# ==============================
X_train, X_test, y_train, y_test, yreg_train, yreg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

# ==============================
# 7️⃣ TREINO CLASSIFICAÇÃO
# ==============================
print("\n===== TREINANDO CLASSIFICAÇÃO =====")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Acurácia Classificação: {acc:.4f}")

# ==============================
# 8️⃣ TREINO REGRESSÃO
# ==============================
print("\n===== TREINANDO REGRESSÃO =====")
reg.fit(X_train, yreg_train)

yreg_pred = reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(yreg_test, yreg_pred))
r2 = r2_score(yreg_test, yreg_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

# ==============================
# 9️⃣ SALVAR MODELOS (PIPELINES COMPLETOS)
# ==============================
os.makedirs("models", exist_ok=True)

joblib.dump(clf, "models/model_class.pkl")
joblib.dump(reg, "models/model_reg.pkl")

print("\n✅ MODELOS SALVOS:")
print(" - models/model_class.pkl")
print(" - models/model_reg.pkl")
print("👉 Próximo passo: subir a API usando esses modelos.")

# python src/TRAIN_READ.py
print("\n📌 Colunas encontradas no Excel:")
print(list(df.columns))