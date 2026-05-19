from __future__ import annotations

import streamlit as st

from components.cards import render_note_card, render_section_header
from components.theme import inject_theme, render_sidebar
from utils.project_content import APP_COPY, DEPLOYMENT_OPTIONS
from utils.state import init_session_state


st.set_page_config(
    page_title="About",
    page_icon=":information_source:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    init_session_state()
    sidebar_state = render_sidebar()
    inject_theme(sidebar_state["theme_name"])

    render_section_header(
        "About This Project",
        "Portfolio-ready positioning, deployment guidance, and honest notes on what is productionized versus notebook-derived.",
    )

    intro_cols = st.columns([1.15, 0.85], gap="large")
    with intro_cols[0]:
        st.markdown(
            """
            This repository demonstrates a practical machine-learning productization workflow:

            - start with exploratory Jupyter notebooks
            - recover the final training logic and model heads
            - package the checkpoints into reusable Python modules
            - present the system through a polished Streamlit interface

            It is especially well suited for:

            - portfolios and job applications
            - hackathon demos
            - ML engineering showcases
            - medical-AI concept demonstrations
            """
        )
    with intro_cols[1]:
        render_note_card(
            "Recommended Positioning",
            "Frame this as a hierarchical computer-vision system for routed blood-cancer image classification using transfer learning.",
        )

    deploy_cols = st.columns(len(DEPLOYMENT_OPTIONS), gap="medium")
    for col, option in zip(deploy_cols, DEPLOYMENT_OPTIONS):
        with col:
            render_note_card(option["name"], option["summary"])
            st.caption(option["detail"])

    detail_cols = st.columns(2, gap="large")
    with detail_cols[0]:
        render_section_header("Free-Tier Deployment Tips", "What to optimize before publishing a public demo.")
        st.markdown(
            """
            - Prefer **CPU inference** if the hosting platform does not provide GPUs
            - Keep model files in the repo only if the host can tolerate the storage size
            - Use `st.cache_resource` to avoid reloading model weights on every rerun
            - Pre-bundle a few example images so recruiters can test the app immediately
            """
        )

    with detail_cols[1]:
        render_section_header("Portfolio Upgrades", "High-impact improvements if you want to take this project further.")
        st.markdown(
            """
            - Add Grad-CAM visual overlays for image explainability
            - Refactor notebook training loops into a configurable training package
            - Add experiment tracking with MLflow or Weights & Biases
            - Deploy a FastAPI backend for serving and keep Streamlit as the UI layer
            """
        )

    render_section_header("Responsible Use", "Important context for a medical-image demo.")
    st.warning(APP_COPY["medical_disclaimer"])


if __name__ == "__main__":
    main()
