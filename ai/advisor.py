import os
import json


def build_prompt(input_data: dict, trend: str, probabilities: dict, simulations: list) -> str:
    sim_text = ""
    for s in simulations:
        sim_text += f"- {s['name']}: tendência '{s['class']}', deltas {s['deltas']}\n"

    return f"""Você é um estrategista de conteúdo digital. Analise os dados abaixo e dê recomendações em português.

## Dados do usuário
- Categoria: {input_data.get('category_name', 'N/A')}
- Plataforma: {input_data.get('platform_name', 'N/A')}
- Região: {input_data.get('region_name', 'N/A')}
- Idioma: {input_data.get('language_name', 'N/A')}
- Porte do criador: {input_data.get('creator_tier_name', 'N/A')}
- Principal fonte de tráfego: {input_data.get('traffic_source_name', 'N/A')}

## Previsão do modelo
- Tendência prevista: {trend}
- Probabilidades: {json.dumps(probabilities, ensure_ascii=False)}

## Simulações
{sim_text}

## Formato da resposta
Responda APENAS com um JSON válido neste formato exato (sem markdown, sem explicações):
{{
  "plataforma_ideal": "...",
  "estrategia_trafego": "...",
  "sugestoes_hashtag": "...",
  "melhor_dia": "...",
  "tom_conteudo": "...",
  "resumo": "..."
}}
"""


def get_recommendations(input_data: dict, trend: str, probabilities: dict, simulations: list) -> dict:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return _fallback(trend)

    prompt = build_prompt(input_data, trend, probabilities, simulations)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600,
        )
        text = resp.choices[0].message.content.strip()
        return json.loads(text)
    except Exception as e:
        return {**_fallback(trend), "erro": str(e)}


def _fallback(trend: str) -> dict:
    tips = {
        "rising": "Invista agora! O conteúdo está em alta. Aproveite o momento com postagens frequentes e hashtags populares.",
        "stable": "Mantenha a consistência. O conteúdo tem desempenho estável. Foque em engajamento para impulsionar.",
        "declining": "Revise a estratégia. O interesse está caindo. Tente novos formatos ou abordagens.",
        "seasonal": "Aproveite a sazonalidade. Planeje postagens nos picos de interesse identificados.",
    }
    return {
        "plataforma_ideal": "youtube" if trend in ("stable", "seasonal") else "tiktok",
        "estrategia_trafego": "hashtag e feed",
        "sugestoes_hashtag": "#trending #viral #content",
        "melhor_dia": "fim de semana",
        "tom_conteudo": "autêntico e educativo",
        "resumo": tips.get(trend, "Analise os dados para definir a melhor estratégia."),
    }
