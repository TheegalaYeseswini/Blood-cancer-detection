from __future__ import annotations

from pathlib import Path

import streamlit as st


CSS_PATH = Path(__file__).resolve().parents[1] / "assets" / "theme.css"

THEMES = {
    "Midnight": {
        "bg": "#07111F",
        "bg_alt": "#0E1B2E",
        "panel": "rgba(15, 23, 42, 0.78)",
        "panel_alt": "rgba(19, 31, 53, 0.88)",
        "border": "rgba(148, 163, 184, 0.18)",
        "text": "#E8EEF7",
        "muted": "#A5B4C7",
        "accent": "#7A8CFF",
        "accent_soft": "#1B9AAA",
    },
    "Graphite": {
        "bg": "#0C0F14",
        "bg_alt": "#151B24",
        "panel": "rgba(20, 26, 35, 0.82)",
        "panel_alt": "rgba(29, 37, 50, 0.9)",
        "border": "rgba(148, 163, 184, 0.16)",
        "text": "#EDF2F7",
        "muted": "#B8C2D1",
        "accent": "#36D399",
        "accent_soft": "#7A8CFF",
    },
}


def inject_theme(theme_name: str) -> None:
    theme = THEMES.get(theme_name, THEMES["Midnight"])
    css = CSS_PATH.read_text(encoding="utf-8")
    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {theme['bg']};
            --bg-alt: {theme['bg_alt']};
            --panel: {theme['panel']};
            --panel-alt: {theme['panel_alt']};
            --border: {theme['border']};
            --text: {theme['text']};
            --muted: {theme['muted']};
            --accent: {theme['accent']};
            --accent-soft: {theme['accent_soft']};
        }}
        {css}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> dict[str, str]:
    with st.sidebar:
        st.markdown("## Blood Cancer AI")
        st.caption("Interactive clinical-style demo for routed microscopy inference.")
        st.page_link("app.py", label="Overview")
        st.page_link("pages/1_Inference_Studio.py", label="Inference Studio")
        st.page_link("pages/2_Model_Insights.py", label="Model Insights")
        st.page_link("pages/3_Workflow_and_Data.py", label="Workflow & Data")
        st.page_link("pages/4_About.py", label="About")
        st.divider()

        theme_name = st.selectbox(
            "Theme preset",
            list(THEMES.keys()),
            index=list(THEMES.keys()).index(st.session_state.get("theme_name", "Midnight")),
        )
        st.session_state["theme_name"] = theme_name

        device_preference = st.selectbox(
            "Inference device",
            ["auto", "cpu", "cuda"],
            index=["auto", "cpu", "cuda"].index(st.session_state.get("device_preference", "auto")),
        )
        st.session_state["device_preference"] = device_preference

        analyst_name = st.text_input(
            "Analyst label",
            value=st.session_state.get("analyst_name", "Demo Analyst"),
        )
        st.session_state["analyst_name"] = analyst_name

        st.caption(
            f"Prediction history: {len(st.session_state.get('prediction_history', []))} run(s)"
        )

        if st.button("Clear session history", use_container_width=True):
            st.session_state["prediction_history"] = []
            st.session_state["last_result"] = None

    return {
        "theme_name": st.session_state["theme_name"],
        "device_preference": st.session_state["device_preference"],
        "analyst_name": st.session_state["analyst_name"],
    }
