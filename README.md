# Projeto Passos Mágicos
Datathon - Passos Mágicos (MLOps)

1. Visão Geral do Projeto
Este projeto tem como objetivo prever o risco de defasagem escolar de alunos da Associação Passos Mágicos a partir de indicadores educacionais e notas.  
A solução foi construída seguindo boas práticas de Machine Learning Engineering e MLOps, contemplando treinamento do modelo, API para inferência, dashboard para visualização, testes unitários e monitoramento de drift.

2. Problema de Negócio
Identificar alunos com maior risco de defasagem escolar para apoiar intervenções pedagógicas precoces e direcionar melhor os recursos educacionais da instituição.

3. Stack Tecnológica
- Linguagem: Python 3.x
- Machine Learning: scikit-learn, pandas, numpy
- Modelos: RandomForest (Classificação e Regressão)
- API: FastAPI
- Serialização: joblib
- Dashboard: Streamlit e Plotly
- Testes: pytest
- Monitoramento: logs de predição e script de detecção de drift

4. Estrutura do Projeto

project-root/
 ├── src/
 │   ├── TRAIN_READ.py      - Treinamento dos modelos
 │   ├── API.py             - API FastAPI (/predict)
 │   └── STREAMLIT.py       - Dashboard
 ├── tests/                 - Testes unitários (pytest)
 │   └── test_training.py
 ├── monitoring/            - Monitoramento de drift
 │   └── drift_check.py
 ├── logs/                  - Logs de produção da API
 ├── models/                - Modelos treinados (.pkl)
 ├── data/                  - Base de dados (Excel)
 └── README.txt

5. Como Rodar o Projeto Localmente

5.1 Criar ambiente virtual e instalar dependências
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

5.2 Treinar os modelos
python src/TRAIN_READ.py

5.3 Subir a API
uvicorn src.API:app --reload

5.4 Rodar o Dashboard
streamlit run src/STREAMLIT.py

6. Exemplo de Chamada da API

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Ano_nasc": 2007,
    "Idade_22": 15,
    "Ano_ingresso": 2021,
    "Genero": "Menino",
    "Fase": "3",
    "Turma": "B",
    "IAA": 7.5,
    "IEG": 6.0,
    "IPS": 7.5,
    "IDA": 2.7,
    "IPV": 4.7,
    "IAN": 5.0,
    "Matem": 6.0,
    "Portug": 6.0,
    "Ingles": 6.0
  }'

Resposta esperada:
{
  "risco_defasagem": 1,
  "probabilidade_risco": 0.82,
  "defasagem_prevista": -1.3
}

7. Pipeline de Machine Learning

7.1 Pre-processamento dos Dados
Padronização de variáveis numéricas e codificação one-hot para variáveis categóricas.

7.2 Engenharia de Features
Seleção de indicadores educacionais e notas como variáveis de entrada.

7.3 Treinamento e Validação
Uso de RandomForest para classificação (risco) e regressão (nível de defasagem).
Métricas: Accuracy para classificação, RMSE e R2 para regressão.

7.4 Deploy
Exposição dos modelos por meio de uma API FastAPI e consumo via dashboard Streamlit.

7.5 Testes Unitários
Implementação de testes unitários com pytest para validar o pipeline de treinamento e a inferência dos modelos.

7.6 Monitoramento Contínuo
Registro das predições em logs e verificação de drift comparando estatísticas dos dados em produção com um baseline inicial.

8. Observação Importante sobre Interpretação de Risco

O sistema utiliza dois níveis de decisão:
1) Modelo de Machine Learning: prevê o risco de defasagem com base em múltiplos indicadores.
2) Regra pedagógica (interpretação humana): traduz a previsão do modelo em categorias práticas considerando também as notas.

Dessa forma, a decisão final apresentada no dashboard é uma combinação de predição do modelo com uma regra de negócio pedagógica, tornando a solução mais interpretável para educadores.

9. Autores
Projeto desenvolvido para o Datathon - Pós Tech (Machine Learning Engineering).