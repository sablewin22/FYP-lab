from dotenv import load_dotenv
load_dotenv()

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

st.set_page_config(page_title="FYP Lab", layout="centered")
st.title("FYP Lab — Estratégia para TikTok e YouTube Shorts")

COLUMN_MAP = {
    "platform": "platform_cat", "region": "region_cat", "language": "language_cat",
    "category": "category_cat", "traffic_source": "traffic_source_cat", "creator_tier": "creator_tier_cat",
}

CATEGORIES = [
    ("Comida", "Food"), ("Comédia", "Comedy"), ("Tecnologia", "Tech"),
    ("Beleza", "Beauty"), ("Notícias", "News"), ("Educação", "Education"),
    ("Música", "Music"), ("Jogos", "Gaming"), ("Viagem", "Travel"),
    ("Estilo de Vida", "Lifestyle"), ("Esportes", "Sports"),
]

PLATFORMS = [("YouTube", "youtube"), ("TikTok", "tiktok")]

REGIONS = [
    ("África", "Africa"), ("Europa", "Europe"), ("Américas", "Americas"),
    ("Ásia", "Asia"), ("Oceania", "Oceania"),
]

LANGUAGES = [
    ("Inglês", "en"), ("Português", "pt"), ("Espanhol", "es"),
    ("Italiano", "it"), ("Japonês", "ja"), ("Alemão", "de"),
    ("Russo", "ru"), ("Hindi", "hi"), ("Turco", "tr"),
    ("Coreano", "ko"), ("Francês", "fr"), ("Árabe", "ar"),
]

CREATOR_TIERS = [("Nano", "nano"), ("Micro", "micro"), ("Médio", "mid"), ("Macro", "macro")]

TRAFFIC_SOURCES = [
    ("Pesquisa", "search"), ("Feed", "feed"), ("Compartilhamento", "share"),
    ("Perfil", "profile"), ("Externo", "external"), ("Hashtag", "hashtag"),
]

TREND_LABELS = {
    "rising": "Em Alta", "stable": "Estável",
    "declining": "Em Queda", "seasonal": "Sazonal",
}

TREND_COLORS = {
    "rising": "#22c55e", "stable": "#3b82f6",
    "declining": "#ef4444", "seasonal": "#f59e0b",
}

FEATURE_LABELS = {
    "share_rate_log": "Taxa de Compartilhamento",
    "like_rate_log": "Taxa de Curtidas",
    "weekend_hashtag_boost": "Hashtag no Fim de Semana",
    "views_per_day": "Visualizações por Dia",
    "comment_rate_log": "Taxa de Comentários",
    "title_len": "Tamanho do Título",
    "text_richness": "Riqueza do Texto",
    "platform_cat": "Plataforma",
    "region_cat": "Região",
    "language_cat": "Idioma",
    "category_cat": "Categoria",
    "traffic_source_cat": "Origem do Tráfego",
    "creator_tier_cat": "Porte do Criador",
}

@st.cache_data
def load_and_prepare():
    df = get_dataframe()
    mapping = {}
    for raw_col, enc_col in COLUMN_MAP.items():
        pairs = df[[raw_col, enc_col]].drop_duplicates().sort_values(enc_col)
        mapping[raw_col] = dict(zip(pairs[raw_col], pairs[enc_col]))
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
        cat_display = st.selectbox("Categoria", [c[0] for c in CATEGORIES])
        plat_display = st.selectbox("Plataforma", [p[0] for p in PLATFORMS])
        reg_display = st.selectbox("Região", [r[0] for r in REGIONS])
    with col2:
        lang_display = st.selectbox("Idioma", [l[0] for l in LANGUAGES])
        tier_display = st.selectbox("Porte do Criador", [t[0] for t in CREATOR_TIERS])
        traffic_display = st.selectbox("Origem do Tráfego", [t[0] for t in TRAFFIC_SOURCES])
    submitted = st.form_submit_button("Analisar", use_container_width=True)

if submitted:
    cat_raw = dict(CATEGORIES)[cat_display]
    plat_raw = dict(PLATFORMS)[plat_display]
    reg_raw = dict(REGIONS)[reg_display]
    lang_raw = dict(LANGUAGES)[lang_display]
    tier_raw = dict(CREATOR_TIERS)[tier_display]
    traffic_raw = dict(TRAFFIC_SOURCES)[traffic_display]

    base_input = {
        "platform_cat": MAPPING["platform"][plat_raw],
        "region_cat": MAPPING["region"][reg_raw],
        "language_cat": MAPPING["language"][lang_raw],
        "category_cat": MAPPING["category"][cat_raw],
        "traffic_source_cat": MAPPING["traffic_source"][traffic_raw],
        "creator_tier_cat": MAPPING["creator_tier"][tier_raw],
        "title_len": 50, "text_richness": 1.5,
        "like_rate_log": 0.12, "comment_rate_log": 0.08, "share_rate_log": 0.04,
        "views_per_day": 1000, "weekend_hashtag_boost": 0,
    }

    with st.spinner("Analisando..."):
        trend, probs, top5 = predict(base_input)
        scenarios = [
            {"name": "Postar no fim de semana com hashtag", "changes": {"weekend_hashtag_boost": 1}},
            {"name": "Upgrade de criador (Nano -> Médio)", "changes": {"creator_tier_cat": MAPPING["creator_tier"]["mid"]}},
            {"name": "Alto tráfego + hashtag", "changes": {"views_per_day": 50000, "weekend_hashtag_boost": 1}},
        ]
        sim_results, *_ = simulate(base_input, scenarios)

        display_input = {
            "platform_name": plat_display, "region_name": reg_display,
            "language_name": lang_display, "category_name": cat_display,
            "creator_tier_name": tier_display, "traffic_source_name": traffic_display,
        }
        recs = get_recommendations(display_input, trend, probs, sim_results_pt)

    st.subheader("Resultado da Previsão")
    trend_pt = TREND_LABELS.get(trend, trend)
    color = TREND_COLORS.get(trend, "#333")
    st.markdown(f"<h2 style='color:{color}'>{trend_pt}</h2>", unsafe_allow_html=True)

    probs_pt = {TREND_LABELS.get(k, k): v for k, v in probs.items()}
    fig = go.Figure(data=[go.Bar(
        x=list(probs_pt.keys()), y=list(probs_pt.values()),
        marker_color=[TREND_COLORS.get(k, "#888") for k in probs.keys()],
    )])
    fig.update_layout(
        title="Probabilidades por Classe",
        yaxis_title="Probabilidade",
        yaxis=dict(range=[0, 1]), height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Principais Fatores na Decisão"):
        for feat, imp in sorted(top5.items(), key=lambda x: -x[1]):
            label = FEATURE_LABELS.get(feat, feat)
            st.write(f"- **{label}**: {imp:.2%}")

    sim_results_pt = []
    for s in sim_results:
        sim_results_pt.append({
            "name": s["name"],
            "class": TREND_LABELS.get(s["class"], s["class"]),
            "deltas": {TREND_LABELS.get(k, k): v for k, v in s["deltas"].items()},
        })

    st.subheader("Simulação de Cenários")
    for s in sim_results_pt:
        with st.container():
            st.markdown(f"**{s['name']}** — Tendência: `{s['class']}`")
            for cls, delta in s["deltas"].items():
                arrow = "▲" if delta > 0 else "▼" if delta < 0 else "—"
                st.write(f"  {arrow} {cls}: {delta:+.2%}")
            st.divider()

    if recs:
        st.subheader("Recomendações da IA")
        card = f"""
        <div style="background:#1a1a2e;padding:20px;border-radius:12px;color:white;margin:10px 0">
            <p><strong>Plataforma ideal:</strong> {recs.get('plataforma_ideal','—')}</p>
            <p><strong>Estratégia de tráfego:</strong> {recs.get('estrategia_trafego','—')}</p>
            <p><strong>Sugestões de hashtag:</strong> {recs.get('sugestoes_hashtag','—')}</p>
            <p><strong>Melhor dia:</strong> {recs.get('melhor_dia','—')}</p>
            <p><strong>Tom do conteúdo:</strong> {recs.get('tom_conteudo','—')}</p>
            <p><strong>Resumo:</strong> {recs.get('resumo','—')}</p>
        </div>
        """
        st.markdown(card, unsafe_allow_html=True)

        try:
            pdf_bytes = generate_pdf(display_input, trend_pt, probs_pt, sim_results_pt, recs)
            st.download_button(
                "Exportar PDF", data=pdf_bytes, file_name="contentai_report.pdf",
                mime="application/pdf", use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Não foi possível gerar o PDF: {e}")
