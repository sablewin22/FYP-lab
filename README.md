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

O FYP Lab é uma ferramenta que ajuda criadores de conteúdo a decidir o que postar no TikTok e YouTube Shorts. O usuário preenche um formulário com informações como categoria, plataforma, região, idioma e porte do criador, e o sistema prevê se a tendência do conteúdo será Em Alta, Estável, Em Queda ou Sazonal. Um gráfico mostra as probabilidades de cada classe, e uma lista indica quais fatores mais influenciaram a previsão.

O projeto também simula cenários alternativos, como postar no fim de semana com hashtag ou fazer um upgrade de porte de criador, mostrando como a tendência mudaria em cada caso. Além disso, o sistema gera recomendações personalizadas de plataforma ideal, estratégia de tráfego, hashtags, melhor dia e tom do conteúdo usando inteligência artificial. Por fim, é possível exportar um relatório completo em PDF com todos os resultados.

A interface foi construída com Streamlit. O modelo de machine learning é um RandomForestClassifier do scikit-learn, que atinge cerca de 81% de acurácia. Foi treinado com 13 variáveis extraídas de um dataset de 50 mil posts do TikTok e YouTube, disponível no Hugging Face (YouTube/TikTok Trends Dataset 2025). O cache dos dados é feito em Parquet com Pandas e PyArrow. Os gráficos são gerados com Plotly, o PDF com ReportLab e as recomendações por IA utilizam a API da Groq com o modelo Llama 3.3 70B.

O projeto está versionado no GitHub e pode ser rodado localmente com streamlit run app.py após instalar as dependências com pip install -r requirements.txt e treinar o modelo com python -m ml.train. Também é compatível com deploy no Hugging Face Spaces com SDK Streamlit, sendo necessário configurar a chave GROQ_API_KEY nos Secrets do Space.

Equipe: Júlia Cereja e Sabrina Azulay. Orientador: Josir Cardoso Gomes.
