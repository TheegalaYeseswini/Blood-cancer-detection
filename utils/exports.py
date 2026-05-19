from __future__ import annotations

import io
import json

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


def result_to_json_bytes(result: dict[str, object]) -> bytes:
    return json.dumps(result, indent=2).encode("utf-8")


def probabilities_to_csv_bytes(probability_frame: pd.DataFrame) -> bytes:
    return probability_frame.to_csv(index=False).encode("utf-8")


def build_pdf_report(result: dict[str, object], analyst_name: str) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    text = pdf.beginText(18 * mm, height - 20 * mm)
    text.setFont("Helvetica-Bold", 16)
    text.textLine("Blood Cancer AI Prediction Report")
    text.setFont("Helvetica", 10)
    text.textLine("")
    text.textLine(f"Analyst: {analyst_name}")
    text.textLine(f"Source: {result['meta']['source_name']}")
    text.textLine(f"Device: {result['meta']['device']}")
    text.textLine(f"Inference time: {result['meta']['inference_ms']} ms")
    text.textLine("")
    text.textLine("Broad classifier result")
    text.textLine(
        f"- Label: {result['tetraclassifier']['predicted_label']} "
        f"({result['tetraclassifier']['confidence'] * 100:.2f}%)"
    )
    text.textLine("")
    text.textLine("Combined routed result")
    text.textLine(f"- Primary label: {result['combined']['primary_label']}")
    text.textLine(f"- Secondary label: {result['combined']['secondary_label']}")
    text.textLine(f"- Route: {result['combined']['used_subtype_model'] or 'none'}")
    text.textLine("")
    text.textLine("Summary")
    for chunk in _wrap_text(result["combined"]["summary"], limit=90):
        text.textLine(chunk)

    subtype_prediction = result["selected_subtype_model"]
    if subtype_prediction is not None:
        text.textLine("")
        text.textLine("Subtype classifier result")
        text.textLine(
            f"- Label: {subtype_prediction['predicted_label']} "
            f"({subtype_prediction['confidence'] * 100:.2f}%)"
        )

    pdf.drawText(text)
    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer.read()


def _wrap_text(text: str, limit: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    current_length = 0

    for word in words:
        projected = current_length + len(word) + (1 if current else 0)
        if projected > limit:
            lines.append(" ".join(current))
            current = [word]
            current_length = len(word)
        else:
            current.append(word)
            current_length = projected

    if current:
        lines.append(" ".join(current))
    return lines
