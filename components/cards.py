from __future__ import annotations

from html import escape
from typing import Iterable

import streamlit as st


def render_hero(eyebrow: str, title: str, description: str, pills: Iterable[str]) -> None:
    pill_html = "".join(
        f'<span class="pill">{escape(pill)}</span>' for pill in pills
    )
    st.markdown(
        f"""
        <section class="hero-card">
            <div class="eyebrow">{escape(eyebrow)}</div>
            <h1>{escape(title)}</h1>
            <p>{escape(description)}</p>
            <div class="pill-row">{pill_html}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_metric_grid(metrics: list[dict[str, str]]) -> None:
    columns = st.columns(len(metrics), gap="medium")
    for column, metric in zip(columns, metrics):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{escape(metric['label'])}</div>
                    <div class="metric-value">{escape(metric['value'])}</div>
                    <div class="metric-caption">{escape(metric['caption'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_section_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-header">
            <h2>{escape(title)}</h2>
            <p>{escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_note_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="card-title">{escape(title)}</div>
            <div class="card-subtitle">{escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
