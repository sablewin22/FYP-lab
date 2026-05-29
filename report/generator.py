import io
import tempfile
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import plotly.graph_objects as go
from pathlib import Path


def _plot_to_image(probs: dict, w=400, h=250) -> ImageReader:
    fig = go.Figure(data=[go.Bar(x=list(probs.keys()), y=list(probs.values()),
                                  marker_color=["#ef4444", "#22c55e", "#f59e0b", "#3b82f6"])])
    fig.update_layout(title="Probabilidades", xaxis_title="Classe", yaxis_title="Probabilidade",
                      margin=dict(l=20, r=20, t=40, b=30))
    tmp = Path(tempfile.gettempdir()) / "contentai_chart.png"
    fig.write_image(str(tmp), width=w, height=h, scale=2)
    return ImageReader(str(tmp))


def generate_pdf(input_data: dict, trend: str, probs: dict, simulations: list, recommendations: dict) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 30

    c.setFont("Helvetica-Bold", 18)
    c.drawString(30, y, "Relatorio ContentAI")
    y -= 40

    c.setFont("Helvetica-Bold", 12)
    c.drawString(30, y, "Dados do Input")
    c.setFont("Helvetica", 10)
    y -= 16
    for k, v in input_data.items():
        if k.endswith("_name") and v:
            label = k.replace("_name", "").replace("_", " ").title()
            c.drawString(40, y, f"{label}: {v}")
            y -= 14
    y -= 10

    c.setFont("Helvetica-Bold", 12)
    c.drawString(30, y, "Previsao do Modelo")
    c.setFont("Helvetica", 10)
    y -= 16
    c.drawString(40, y, f"Tendencia: {trend}")
    y -= 40

    if probs:
        img = _plot_to_image(probs)
        c.drawImage(img, 40, y - 120, width=360, height=120)
        y -= 140

    c.setFont("Helvetica-Bold", 12)
    c.drawString(30, y, "Cenarios Simulados")
    c.setFont("Helvetica", 10)
    y -= 16
    for s in simulations:
        c.drawString(40, y, f"{s['name']}: {s['class']} | Deltas: {s['deltas']}")
        y -= 14
    y -= 10

    c.setFont("Helvetica-Bold", 12)
    c.drawString(30, y, "Recomendacoes da IA")
    c.setFont("Helvetica", 10)
    y -= 16
    for k, v in recommendations.items():
        if k != "erro":
            label = k.replace("_", " ").title()
            text = f"{label}: {v}"
            c.drawString(40, y, text[:90])
            y -= 14

    c.save()
    buf.seek(0)
    return buf.getvalue()
