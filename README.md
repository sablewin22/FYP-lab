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

# ContentAI

Ferramenta inteligente para criadores de conteúdo decidirem o que postar no TikTok e YouTube Shorts.

## Instalação

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Treinar o modelo (primeira vez)
python -m ml.train

# Rodar a interface web
streamlit run app.py
```

## Deploy no Hugging Face Spaces

1. Crie um Space em https://huggingface.co/new-space
2. Escolha Streamlit SDK
3. Faça push do repositório
4. Adicione `GROQ_API_KEY` nos Secrets do Space

## Estrutura

```
contentai/
├── app.py              # Interface Streamlit
├── data/loader.py      # Download e cache do dataset
├── ml/preprocess.py    # Pré-processamento
├── ml/train.py         # Treino do modelo
├── ml/predict.py       # Inferência
├── ml/simulator.py     # Simulação de cenários
├── ai/advisor.py       # Recomendações via IA
└── report/generator.py # Geração de PDF
```
