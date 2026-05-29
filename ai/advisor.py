import os
import json


def build_prompt(input_data: dict, trend: str, probabilities: dict, simulations: list) -> str:
    sim_text = ""
    for s in simulations:
        deltas_text = ", ".join(f"{k}: {v:+.0%}" for k, v in s["deltas"].items())
        sim_text += f"- {s['name']}: tendência '{s['class']}', impacto {deltas_text}\n"

    return f"""Você é um estrategista de conteúdo digital. Analise os dados abaixo e dê recomendações personalizadas em português, considerando as características específicas do usuário.

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

## Instruções
- Sua resposta deve ser DIFERENTE para cada combinação de categoria, plataforma e tendência.
- Sugira hashtags relevantes para a categoria específica do usuário.
- O melhor dia deve ser adaptado ao nicho.
- O tom do conteúdo deve fazer sentido para a categoria (ex: Tech = educativo, Comédia = descontraído).

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
        return _fallback(input_data, trend, probabilities)

    prompt = build_prompt(input_data, trend, probabilities, simulations)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=600,
        )
        text = resp.choices[0].message.content.strip()
        return json.loads(text)
    except Exception as e:
        return {**_fallback(input_data, trend, probabilities), "erro": str(e)}


def _fallback(input_data: dict, trend: str, probabilities: dict) -> dict:
    cat = input_data.get("category_name", "")
    plat = input_data.get("platform_name", "")
    traffic = input_data.get("traffic_source_name", "")
    tier = input_data.get("creator_tier_name", "")

    tips = {
        "rising": "Invista agora! O conteúdo está em alta. Aproveite o momento com postagens frequentes.",
        "stable": "Mantenha a consistência. O conteúdo tem desempenho estável. Foque em engajamento.",
        "declining": "Revise a estratégia. O interesse está caindo. Tente novos formatos ou abordagens.",
        "seasonal": "Aproveite a sazonalidade. Planeje postagens nos picos de interesse identificados.",
    }

    hashtags_por_categoria = {
        "Comida": "#receita #comida #gastronomia #culinária",
        "Comédia": "#humor #comédia #risos #entretenimento",
        "Tecnologia": "#tech #tecnologia #inovação #dicas",
        "Beleza": "#beleza #maquiagem #skincare #tutorial",
        "Notícias": "#notícias #informação #atualidades",
        "Educação": "#educação #aprender #conhecimento #dicas",
        "Música": "#música #artista #lançamento #playlist",
        "Jogos": "#games #jogos #gameplay #gamer",
        "Viagem": "#viagem #turismo #aventura #destinos",
        "Esportes": "#esportes #treino #atleta #motivação",
        "Estilo de Vida": "#estilodevida #bemestar #rotina",
    }

    tom_por_categoria = {
        "Comida": "Apetitoso e instructivo",
        "Comédia": "Descontraído e engraçado",
        "Tecnologia": "Educativo e objetivo",
        "Beleza": "Tutorial e inspirador",
        "Notícias": "Direto e informativo",
        "Educação": "Didático e acessível",
        "Música": "Criativo e envolvente",
        "Jogos": "Divertido e dinâmico",
        "Viagem": "Inspirador e aventureiro",
        "Esportes": "Motivacional e enérgico",
        "Estilo de Vida": "Leve e inspirador",
    }

    trend_pt = {"rising": "Em Alta", "stable": "Estável", "declining": "Em Queda", "seasonal": "Sazonal"}.get(trend, trend)

    hashtag = hashtags_por_categoria.get(cat, "#conteúdo #criador #trend")
    tom = tom_por_categoria.get(cat, "Autêntico e original")
    plataforma = plat if plat else "YouTube"
    melhor_dia = "Sábado" if cat in ("Comédia", "Música", "Jogos") else "Domingo" if cat in ("Educação", "Notícias") else "Sexta-feira"

    if traffic == "Hashtag":
        estrategia = "Investir em hashtags nichadas e trending topics"
    elif traffic == "Feed":
        estrategia = "Produzir conteúdo frequente para alimentar o feed"
    elif traffic == "Pesquisa":
        estrategia = "Otimizar título e descrição com SEO"
    elif traffic == "Compartilhamento":
        estrategia = "Criar conteúdo compartilhável e com apelo viral"
    elif traffic == "Perfil":
        estrategia = "Fortalecer marca pessoal e identidade visual"
    else:
        estrategia = "Combinar hashtags, feed e compartilhamento"

    if tier == "Nano":
        estrategia += " — comece com nichos menores e consistência"
    elif tier == "Micro":
        estrategia += " — foque em engajamento da comunidade"
    elif tier == "Médio":
        estrategia += " — diversifique formatos e colabore"
    elif tier == "Macro":
        estrategia += " — explore parcerias e cross-platform"

    resumo = f"{tips.get(trend, '')} Categoria: {cat}. Plataforma sugerida: {plataforma}."

    return {
        "plataforma_ideal": plataforma,
        "estrategia_trafego": estrategia,
        "sugestoes_hashtag": hashtag,
        "melhor_dia": melhor_dia,
        "tom_conteudo": tom,
        "resumo": resumo,
    }
