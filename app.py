from __future__ import annotations

import streamlit as st

from components.cards import render_hero, render_metric_grid, render_note_card, render_section_header
from components.charts import dataset_distribution_chart, model_accuracy_chart
from components.theme import inject_theme, render_sidebar
from utils.project_content import (
    APP_COPY,
    DATASET_DISTRIBUTION,
    DEPLOYMENT_OPTIONS,
    MODEL_BENCHMARKS,
    SAMPLE_EXPECTATIONS,
    get_sample_catalog,
)
from utils.state import init_session_state


st.set_page_config(
    page_title="Blood Cancer AI",
    page_icon=":drop_of_blood:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_homepage() -> None:
    init_session_state()
    sidebar_state = render_sidebar()
    inject_theme(sidebar_state["theme_name"])

    sample_count = len(get_sample_catalog())
    render_hero(
        eyebrow="Medical AI Showcase",
        title="Hierarchical Blood Cancer Detection Dashboard",
        description=(
            "A Streamlit frontend for routed microscopic cell-image inference. "
            "The app first predicts the broad blood-cancer category, then selectively "
            "runs the relevant subtype model for leukemia or lymphoma."
        ),
        pills=[
            "EfficientNet-B0 + DenseNet121",
            "Microscopy Image Classification",
            "Portfolio-Ready Demo",
        ],
    )

    render_metric_grid(
        [
            {
                "label": "Broad Classifier",
                "value": "99.79%",
                "caption": "Reported test accuracy from notebook evaluation",
            },
            {
                "label": "Subtype Models",
                "value": "2",
                "caption": "Specialized routing for leukemia and lymphoma",
            },
            {
                "label": "Sample Inputs",
                "value": str(sample_count),
                "caption": "Curated test images bundled with the repo",
            },
            {
                "label": "Supported Formats",
                "value": "JPG / PNG / BMP",
                "caption": "Any Pillow-readable image can be uploaded",
            },
        ]
    )

    col_left, col_right = st.columns([1.25, 1], gap="large")
    with col_left:
        render_section_header(
            "What This App Does",
            "The frontend is aligned to the actual repository workflow rather than a generic demo shell.",
        )
        st.markdown(
            """
            - Runs the **tetra disease classifier** first to predict `LEUKEMIA`, `LYMPHOMA`, `MYELOMA`, or `HEALTHY`
            - Routes `LEUKEMIA` predictions to the **4-class leukemia subtype model**
            - Routes `LYMPHOMA` predictions to the **3-class lymphoma subtype model**
            - Exposes notebook-derived metrics, dataset composition, and example imagery in a presentation-ready format
            """
        )

        render_section_header(
            "Interactive Workflow",
            "Use the multipage navigation in the sidebar to explore the project from different angles.",
        )
        st.markdown(
            """
            1. Visit **Inference Studio** to upload an image, use the webcam, or try bundled examples  
            2. Open **Model Insights** to inspect benchmark metrics and architecture choices  
            3. Review **Workflow & Data** to understand routing logic and dataset composition  
            4. Use **About** for deployment notes, limitations, and portfolio positioning  
            """
        )

    with col_right:
        render_section_header(
            "Project Benchmarks",
            "Notebook metrics extracted from the repo and presented as portfolio-friendly summaries.",
        )
        st.plotly_chart(model_accuracy_chart(MODEL_BENCHMARKS), use_container_width=True)

        render_note_card(
            "Deployment Ready",
            "This app is organized for Streamlit Cloud, Hugging Face Spaces, Render, or Docker-based deployment.",
        )

    sample_cols = st.columns(2, gap="large")
    with sample_cols[0]:
        render_section_header(
            "Dataset Composition",
            "The broad classifier blends multiple microscopy datasets into a single routed diagnosis flow.",
        )
        st.plotly_chart(
            dataset_distribution_chart(DATASET_DISTRIBUTION),
            use_container_width=True,
        )

    with sample_cols[1]:
        render_section_header(
            "Bundled Example Inputs",
            "These repository images are useful for quick smoke tests and demos.",
        )
        for sample in get_sample_catalog():
            expected = SAMPLE_EXPECTATIONS.get(sample["name"], "Sample image")
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="card-title">{sample['name']}</div>
                    <div class="card-subtitle">{expected}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    deploy_cols = st.columns(len(DEPLOYMENT_OPTIONS), gap="medium")
    for col, option in zip(deploy_cols, DEPLOYMENT_OPTIONS):
        with col:
            render_note_card(option["name"], option["summary"])
            st.caption(option["detail"])

    render_section_header(
        "Quick Start",
        "Run the Streamlit app locally or fall back to the preserved CLI module for terminal-based inference.",
    )
    st.code(
        "python -m venv venv\n"
        ".\\venv\\Scripts\\Activate.ps1\n"
        "python -m pip install -r requirements.txt\n"
        "streamlit run app.py",
        language="powershell",
    )
    st.caption(APP_COPY["medical_disclaimer"])


if __name__ == "__main__":
    render_homepage()
