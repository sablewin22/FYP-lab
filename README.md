---
title: ContentAI
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.33.0
app_file: app.py
pinned: false
---

# FYP Lab — ContentAI

## O que é o projeto?

Ferramenta inteligente que ajuda criadores de conteúdo a decidir **o que postar** no TikTok e YouTube Shorts. O usuário informa o tema, plataforma, região e porte do criador, e o sistema prevê a tendência do conteúdo, simula cenários alternativos e dá recomendações estratégicas.

## Funcionalidades

- **Previsão de tendência** — classifica o conteúdo em: **Em Alta**, **Estável**, **Em Queda** ou **Sazonal**
- **Gráfico de probabilidades** — mostra a chance de cada classe com Plotly
- **Fatores importantes** — exibe quais variáveis mais influenciaram a decisão
- **Simulação de cenários** — testa "e se" (ex: postar no fim de semana com hashtag, upgrade de criador)
- **Recomendações por IA** — sugestões personalizadas de plataforma, hashtags, tom e dia ideais (via Groq/Llama 3)
- **Exportar PDF** — relatório completo para download

## Stack utilizada

| Tecnologia | Função |
|---|---|
| **Streamlit** | Interface web (frontend + backend) |
| **scikit-learn** | RandomForestClassifier (modelo de ML) |
| **Pandas / PyArrow** | Manipulação e cache dos dados |
| **Plotly** | Gráfico interativo de probabilidades |
| **ReportLab** | Geração de PDF |
| **Groq API (Llama 3.3 70B)** | Recomendações por IA |
| **Hugging Face Datasets** | Fonte dos dados |
| **Git / GitHub** | Versionamento e deploy |

## Onde está rodando?

- **Local**: `streamlit run app.py`
- **Hugging Face Spaces**: deploy com SDK Streamlit

## Dataset

[YouTube/TikTok Trends Dataset 2025](https://huggingface.co/datasets/tarekmasryo/youtube-tiktok-trends-dataset-2025)

- **50.000 posts** de TikTok e YouTube
- Colunas: categoria, plataforma, região, idioma, engajamento (curtidas, comentários, compartilhamentos), porte do criador, fonte de tráfego, etc.
- 4 classes de tendência: rising, stable, declining, seasonal

## Modelo de ML

- **Algoritmo**: RandomForestClassifier (100 árvores)
- **Features**: 13 variáveis (categóricas encoded + numéricas normalizadas)
- **Acurácia**: ~81%
- **Biblioteca**: scikit-learn com joblib para serialização

## Equipe

**Júlia Cereja e Sabrina Azulay**  
Orientador: **Josir Cardoso Gomes**

## Instalação

```bash
pip install -r requirements.txt
python -m ml.train
streamlit run app.py
```

## Deploy no Hugging Face Spaces

1. Crie um Space em https://huggingface.co/new-space (SDK Streamlit)
2. Faça push do repositório
3. Adicione `GROQ_API_KEY` nos Secrets do Space
