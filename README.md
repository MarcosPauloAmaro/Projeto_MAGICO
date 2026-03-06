PROJETO: SISTEMA DE PREVISÃO DE DEFASAGEM ESCOLAR
Autor: Marcos Paulo Amaro

=================================================
1. VISÃO GERAL DO PROJETO
=================================================

Este projeto implementa um pipeline completo de Machine Learning para prever
o risco de defasagem escolar de alunos utilizando indicadores educacionais.

A solução foi desenvolvida seguindo práticas de Machine Learning Engineering
e MLOps, incluindo:

- Treinamento de modelos
- API para inferência
- Dashboard para visualização
- Testes unitários
- Monitoramento de drift
- Containerização com Docker

O sistema permite analisar dados de alunos e prever:

1) Se o aluno apresenta risco de defasagem
2) A probabilidade desse risco
3) O nível estimado de defasagem

=================================================
2. ARQUITETURA DO SISTEMA
=================================================

A arquitetura do projeto é composta por três camadas principais:

1) Treinamento do modelo
2) API de inferência
3) Dashboard de visualização

Fluxo do sistema:

Dados Excel
     ↓
Pipeline de treinamento (Scikit-Learn)
     ↓
Modelos serializados (joblib)
     ↓
API FastAPI
     ↓
Dashboard Streamlit
     ↓
Usuário final

=================================================
3. ESTRUTURA DO PROJETO
=================================================

project-root/

src/
    TRAIN_READ.py
    API.py
    STREAMLIT.py

tests/
    test_training.py

monitoring/
    drift_check.py

models/
    model_class.pkl
    model_reg.pkl

data/
    DADOS.xlsx

logs/
    predict_logs.jsonl

Dockerfile
requirements.txt
README.txt

=================================================
4. PIPELINE DE TREINAMENTO
=================================================

Arquivo: src/TRAIN_READ.py

Responsável por:

1) Carregar os dados do Excel
2) Realizar limpeza e pré-processamento
3) Criar variável alvo de risco
4) Treinar dois modelos
5) Salvar os modelos treinados

Etapas principais:

4.1 Carregamento dos dados

O script lê todas as abas do Excel e concatena em um único DataFrame.

pd.read_excel(sheet_name=None)

4.2 Seleção de features

Variáveis numéricas:

Ano nasc
Idade 22
Ano ingresso
IAA
IEG
IPS
IDA
IPV
IAN
Matem
Portug
Ingles

Variáveis categóricas:

Genero
Fase
Turma

Target de regressão:

Defasagem

4.3 Criação da variável de classificação

Risco = (Defasagem > 0)

Isso transforma o problema em classificação binária.

4.4 Pré-processamento

Utiliza ColumnTransformer para:

- StandardScaler em variáveis numéricas
- OneHotEncoder em variáveis categóricas

4.5 Modelos utilizados

Classificação:
RandomForestClassifier

Regressão:
RandomForestRegressor

4.6 Split de dados

train_test_split(test_size=0.2)

4.7 Métricas avaliadas

Classificação:
accuracy_score

Regressão:
RMSE
R2 score

4.8 Serialização dos modelos

Os modelos são salvos utilizando joblib:

models/model_class.pkl
models/model_reg.pkl

=================================================
5. API DE PREDIÇÃO
=================================================

Arquivo: src/API.py

Framework utilizado: FastAPI

A API expõe um endpoint principal:

POST /predict

Entrada esperada:

{
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
}

Saída da API:

{
 "risco_defasagem": 1,
 "probabilidade_risco": 0.82,
 "defasagem_prevista": -1.3
}

A API realiza:

1) Carregamento dos modelos
2) Transformação dos dados
3) Predição de classificação
4) Predição de regressão
5) Retorno em JSON

=================================================
6. DASHBOARD STREAMLIT
=================================================

Arquivo: src/STREAMLIT.py

Responsável pela interface visual do sistema.

Funcionalidades:

- Visualização geral dos alunos
- Distribuição de defasagem
- Ranking dos melhores alunos
- Consulta individual por RA
- Exibição de indicadores educacionais
- Visualização do nível de risco

O dashboard se comunica com a API via HTTP utilizando requests.

=================================================
7. TESTES UNITÁRIOS
=================================================

Framework: pytest

Arquivo: tests/test_training.py

Testes implementados:

1) Verificar se o treinamento gera modelos

assert os.path.exists("models/model_class.pkl")

2) Verificar se os modelos conseguem realizar previsões

pred_class = clf.predict(df)
pred_reg = reg.predict(df)

Isso garante que o pipeline de treinamento está funcionando corretamente.

=================================================
8. MONITORAMENTO DE DRIFT
=================================================

Arquivo: monitoring/drift_check.py

Função do script:

Comparar estatísticas dos dados em produção com um baseline inicial.

Passos:

1) Ler logs das predições
2) Calcular média das features
3) Comparar com baseline
4) Identificar possíveis mudanças no perfil dos dados

Se a diferença ultrapassar um limite, é sinalizado:

POSSIVEL DRIFT

Isso indica que o modelo pode precisar ser re-treinado.

=================================================
9. LOGS DE PRODUÇÃO
=================================================

Arquivo:

logs/predict_logs.jsonl

Cada chamada da API registra:

- dados de entrada
- timestamp
- resultado da predição

Esses logs são utilizados para monitoramento.

=================================================
10. CONTAINERIZAÇÃO COM DOCKER
=================================================

Arquivo: Dockerfile

O Docker permite empacotar a aplicação em um ambiente isolado.

Etapas do Dockerfile:

1) Usar imagem base Python
2) Instalar dependências
3) Copiar código da aplicação
4) Copiar modelos treinados
5) Executar API com Uvicorn

Comando para build da imagem:

docker build -t projeto-magico-api .

Comando para rodar o container:

docker run -p 8000:8000 projeto-magico-api

A API ficará disponível em:

http://localhost:8000/docs

=================================================
11. EXECUÇÃO DO PROJETO
=================================================

Treinar modelo:

python src/TRAIN_READ.py

Rodar API local:

uvicorn src.API:app --reload

Rodar dashboard:

streamlit run src/STREAMLIT.py

Rodar testes:

pytest

Rodar monitoramento de drift:

python monitoring/drift_check.py

Rodar via Docker:

docker build -t projeto-magico-api .
docker run -p 8000:8000 projeto-magico-api

=================================================
12. CONSIDERAÇÕES FINAIS
=================================================

O projeto demonstra um pipeline completo de Machine Learning,
incluindo treinamento, inferência, visualização, testes,
monitoramento e containerização.

Essa abordagem segue boas práticas de engenharia de Machine Learning
e facilita manutenção, escalabilidade e reprodutibilidade do sistema.
