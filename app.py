import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data.loader import get_dataframe
from ml.preprocess import preprocess, ALL_FEATURES, NUM_FEATURES, PREPROCESSOR_PATH
from ml.train import train, MODEL_PATH
from ml.predict import predict
from ml.simulator import simulate
from ai.advisor import get_recommendations
from report.generator import generate_pdf

st.set_page_config(page_title="ContentAI", layout="centered")
st.title("ContentAI — Estrategia para TikTok e YouTube Shorts")

COLUMN_MAP = {
    "platform": "platform_cat", "region": "region_cat", "language": "language_cat",
    "category": "category_cat", "traffic_source": "traffic_source_cat", "creator_tier": "creator_tier_cat",
}

@st.cache_data
def load_and_prepare():
    df = get_dataframe()
    mapping = {}
    for raw_col, enc_col in COLUMN_MAP.items():
        pairs = df[[raw_col, enc_col]].drop_duplicates().sort_values(enc_col)
        mapping[raw_col] = dict(zip(pairs[raw_col], pairs[enc_col]))
        mapping[f"{raw_col}_reverse"] = dict(zip(pairs[enc_col], pairs[raw_col]))
    return df, mapping

df, MAPPING = load_and_prepare()

@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test, classes = preprocess(df, fit=True)
    model, acc = train(X_train, y_train, X_test, y_test, classes)
    return model, acc, classes

if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
    with st.spinner("Treinando modelo pela primeira vez..."):
        train_model()

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Categoria", sorted(MAPPING["category"].keys()))
        platform = st.selectbox("Plataforma", sorted(MAPPING["platform"].keys()))
        region = st.selectbox("Regiao", sorted(MAPPING["region"].keys()))
    with col2:
        language = st.selectbox("Idioma", sorted(MAPPING["language"].keys()))
        tier = st.selectbox("Porte do criador", sorted(MAPPING["creator_tier"].keys()))
        traffic = st.selectbox("Fonte de trafego", sorted(MAPPING["traffic_source"].keys()))
    submitted = st.form_submit_button("Analisar", use_container_width=True)

if submitted:
    base_input = {
        "platform_cat": MAPPING["platform"][platform],
        "region_cat": MAPPING["region"][region],
        "language_cat": MAPPING["language"][language],
        "category_cat": MAPPING["category"][category],
        "traffic_source_cat": MAPPING["traffic_source"][traffic],
        "creator_tier_cat": MAPPING["creator_tier"][tier],
        "title_len": 50, "text_richness": 1.5,
        "like_rate_log": 0.12, "comment_rate_log": 0.08, "share_rate_log": 0.04,
        "views_per_day": 1000, "weekend_hashtag_boost": 0,
    }

    with st.spinner("Analisando..."):
        trend, probs, top5 = predict(base_input)
        scenarios = [
            {"name": "Postar no fim de semana com hashtag", "changes": {"weekend_hashtag_boost": 1}},
            {"name": "Upgrade de creator tier (nano -> mid)", "changes": {"creator_tier_cat": MAPPING["creator_tier"]["mid"]}},
            {"name": "Alto trafego + hashtag", "changes": {"views_per_day": 50000, "weekend_hashtag_boost": 1}},
        ]
        sim_results, *_ = simulate(base_input, scenarios)

        display_input = {
            "platform_name": platform, "region_name": region, "language_name": language,
            "category_name": category, "creator_tier_name": tier, "traffic_source_name": traffic,
        }
        recs = get_recommendations(display_input, trend, probs, sim_results)

    st.subheader("Resultado da Previsao")
    colors = {"rising": "#22c55e", "stable": "#3b82f6", "declining": "#ef4444", "seasonal": "#f59e0b"}
    st.markdown(f"<h2 style='color:{colors.get(trend,\"#333\")}'>{trend.upper()}</h2>", unsafe_allow_html=True)

    fig = go.Figure(data=[go.Bar(x=list(probs.keys()), y=list(probs.values()),
                                  marker_color=[colors.get(k, "#888") for k in probs.keys()])])
    fig.update_layout(title="Probabilidades por classe", yaxis_title="Probabilidade",
                      yaxis=dict(range=[0, 1]), height=350)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Top 5 features mais importantes"):
        for feat, imp in sorted(top5.items(), key=lambda x: -x[1]):
            st.write(f"- **{feat}**: {imp:.4f}")

    st.subheader("Simulacao de Cenarios")
    for s in sim_results:
        with st.container():
            st.markdown(f"**{s['name']}** — Tendencia: `{s['class']}`")
            for cls, delta in s["deltas"].items():
                arrow = "▲" if delta > 0 else "▼" if delta < 0 else "—"
                st.write(f"  {arrow} {cls}: {delta:+.2%}")
            st.divider()

    if recs:
        st.subheader("Recomendacoes da IA")
        card = f"""
        <div style="background:#1a1a2e;padding:20px;border-radius:12px;color:white;margin:10px 0">
            <p><strong>Plataforma ideal:</strong> {recs.get('plataforma_ideal','—')}</p>
            <p><strong>Estrategia de trafego:</strong> {recs.get('estrategia_trafego','—')}</p>
            <p><strong>Sugestoes de hashtag:</strong> {recs.get('sugestoes_hashtag','—')}</p>
            <p><strong>Melhor dia:</strong> {recs.get('melhor_dia','—')}</p>
            <p><strong>Tom do conteudo:</strong> {recs.get('tom_conteudo','—')}</p>
            <p><strong>Resumo:</strong> {recs.get('resumo','—')}</p>
        </div>
        """
        st.markdown(card, unsafe_allow_html=True)

        pdf_bytes = generate_pdf(display_input, trend, probs, sim_results, recs)
        st.download_button("Exportar PDF", data=pdf_bytes, file_name="contentai_report.pdf",
                           mime="application/pdf", use_container_width=True)
