# src/STREAMLIT.py
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
import math

CAMINHO_DADOS = "data/DADOS.xlsx"
URL_API = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Passos Mágicos • Painel Educacional", layout="wide")

@st.cache_data
def carregar_dados():
    abas = pd.read_excel(CAMINHO_DADOS, sheet_name=None)
    return pd.concat(abas.values(), ignore_index=True)

df = carregar_dados()

def seguro(v):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return 0.0
        return float(v)
    except:
        return 0.0

# ==============================
# SIDEBAR – LEGENDA (MODELO)
# ==============================
with st.sidebar:
    st.title("🧭 Como o Modelo funciona aqui")
    st.markdown("""
**Fases do Modelo neste sistema (em linguagem simples):**

**1️⃣ Treinamento (feito antes de você usar o painel)**  
O modelo foi treinado com dados históricos de alunos (notas, indicadores e defasagem real).  
Ele aprendeu padrões do tipo:  
> “Perfis parecidos com este costumam ter X nível de defasagem.”

**2️⃣ Previsão (acontece agora)**  
Quando você busca um aluno, o sistema envia os dados dele para o modelo,  
e o modelo devolve **duas respostas**:
- 🤖 Se há risco (Sim/Não)  
- 📐 Qual o nível de defasagem (número de -5 a 2)

**3️⃣ Interpretação pedagógica (regra humana)**  
O painel aplica uma **regra pedagógica** por cima do modelo para ajudar na decisão prática:
- 🔴 Alto risco: defasagem negativa **e** nota < 6  
- 🟠 Risco: defasagem negativa  
- 🟢 Sem risco: defasagem ≥ 0  

> Ou seja: o modelo sugere, e o painel traduz isso para ações pedagógicas.
    """)

st.title("🎓 Painel Educacional — Risco de Defasagem (Passos Mágicos)")
st.caption("Dashboard geral + análise individual de alunos")

# ==============================
# VISÃO GERAL
# ==============================
st.subheader("📊 Visão Geral dos Alunos")

col_g1, col_g2 = st.columns(2)

with col_g1:
    st.markdown("**Distribuição da Defasagem (quanto maior, melhor)**")
    fig_dist = px.histogram(df, x="Defasagem", nbins=20)
    st.plotly_chart(fig_dist, use_container_width=True)

with col_g2:
    st.markdown("**🏆 Top 20 melhores alunos (maior defasagem = melhor)**")
    top20 = (
        df[["RA", "Nome", "Turma", "Fase", "Defasagem"]]
        .dropna(subset=["Defasagem"])
        .sort_values("Defasagem", ascending=False)
        .head(20)
    )
    st.dataframe(top20, use_container_width=True, hide_index=True)

st.markdown("---")

# ==============================
# CONSULTA INDIVIDUAL
# ==============================
st.subheader("🔎 Consulta individual do aluno")
ra = st.text_input("Digite o RA do aluno para análise detalhada")
buscar = st.button("Buscar aluno")

if buscar and ra:
    aluno = df[df["RA"] == ra]
    if aluno.empty:
        st.error("❌ RA não encontrado.")
        st.stop()
    aluno = aluno.iloc[0]

    # Perfil
    st.subheader("👤 Perfil do Aluno")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Nome", aluno.get("Nome", "—"))
    p2.metric("Turma", aluno.get("Turma", "—"))
    p3.metric("Fase", aluno.get("Fase", "—"))
    p4.metric("Gênero", aluno.get("Gênero", "—"))

    # Indicadores
    st.subheader("📊 Indicadores Educacionais")
    i1, i2, i3, i4, i5, i6 = st.columns(6)
    i1.metric("IAA", seguro(aluno.get("IAA")))
    i2.metric("IEG", seguro(aluno.get("IEG")))
    i3.metric("IPS", seguro(aluno.get("IPS")))
    i4.metric("IDA", seguro(aluno.get("IDA")))
    i5.metric("IPV", seguro(aluno.get("IPV")))
    i6.metric("IAN", seguro(aluno.get("IAN")))

    # Notas
    nota_mat = seguro(aluno.get("Matem"))
    nota_por = seguro(aluno.get("Portug"))
    nota_ing = seguro(aluno.get("Ingles"))

    st.subheader("📚 Notas do Aluno")
    n1, n2, n3 = st.columns(3)
    n1.metric("Matemática", nota_mat)
    n2.metric("Português", nota_por)
    n3.metric("Inglês", nota_ing)

    dados_api = {
        "Ano_nasc": seguro(aluno.get("Ano nasc")),
        "Idade_22": seguro(aluno.get("Idade 22")),
        "Ano_ingresso": int(seguro(aluno.get("Ano ingresso"))),
        "Genero": str(aluno.get("Gênero")),
        "Fase": str(aluno.get("Fase")),
        "Turma": str(aluno.get("Turma")),
        "IAA": seguro(aluno.get("IAA")),
        "IEG": seguro(aluno.get("IEG")),
        "IPS": seguro(aluno.get("IPS")),
        "IDA": seguro(aluno.get("IDA")),
        "IPV": seguro(aluno.get("IPV")),
        "IAN": seguro(aluno.get("IAN")),
        "Matem": nota_mat,
        "Portug": nota_por,
        "Ingles": nota_ing,
    }

    with st.spinner("🤖 Consultando os modelos..."):
        resp = requests.post(URL_API, json=dados_api, timeout=15)
        if resp.status_code != 200:
            st.error(f"Erro na API ({resp.status_code})")
            st.code(resp.text)
            st.stop()
        resultado = resp.json()

    classe = resultado.get("risco_defasagem")
    prob = resultado.get("probabilidade_risco")
    defasagem = resultado.get("defasagem_prevista")

    # ==============================
    # AVALIAÇÃO FINAL
    # ==============================
    st.markdown("---")
    st.subheader("🧠 Avaliação Final de Risco do Aluno")

    notas_baixas = any(n < 6 for n in [nota_mat, nota_por, nota_ing])
    if defasagem < 0 and notas_baixas:
        nivel_risco, emoji = "ALTO RISCO", "🔴"
        explicacao = "Defasagem negativa e pelo menos uma nota abaixo de 6."
    elif defasagem < 0:
        nivel_risco, emoji = "RISCO", "🟠"
        explicacao = "Defasagem negativa."
    else:
        nivel_risco, emoji = "SEM RISCO", "🟢"
        explicacao = "Defasagem maior ou igual a 0."

    st.metric("Nível de risco (final)", f"{emoji} {nivel_risco}")
    st.caption(f"Motivo: {explicacao}")

    # ==============================
    # MODELO 2 — REGRESSÃO
    # ==============================
    st.markdown("---")
    st.subheader("📐 Nível de Defasagem (Regressão)")
    st.caption("Escala: -5 (pior) → 2 (melhor). Quanto maior, melhor.")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Defasagem prevista", f"{defasagem:.2f}")
    r2.metric("Matemática", nota_mat)
    r3.metric("Português", nota_por)
    r4.metric("Inglês", nota_ing)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=defasagem,
        title={'text': "Nível de defasagem escolar"},
        gauge={
            'axis': {'range': [-5, 2]},
            'steps': [
                {'range': [-5, -2], 'color': '#ffcccc'},
                {'range': [-2, 0], 'color': '#fff0b3'},
                {'range': [0, 2], 'color': '#ccffcc'},
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # ==============================
    # RECOMENDAÇÕES
    # ==============================
    st.markdown("---")
    st.subheader("📝 Recomendações Sugeridas")

    recomendacoes = []
    if defasagem < 0:
        recomendacoes.append("Acompanhar o aluno de perto nas próximas avaliações.")
    if nota_mat < 6:
        recomendacoes.append("Sugerir reforço em Matemática.")
    if nota_por < 6:
        recomendacoes.append("Sugerir reforço em Português.")
    if nota_ing < 6:
        recomendacoes.append("Sugerir reforço em Inglês.")

    if recomendacoes:
        for r in recomendacoes:
            st.write(f"• {r}")
    else:
        st.success("Nenhuma ação imediata sugerida. Manter acompanhamento regular.")

# streamlit run src/STREAMLIT.py
# streamlit run src/STREAMLIT.py
