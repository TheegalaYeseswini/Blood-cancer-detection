from __future__ import annotations

import streamlit as st

from components.cards import render_note_card, render_section_header
from components.charts import dataset_distribution_chart
from components.theme import inject_theme, render_sidebar
from utils.project_content import APP_COPY, DATASET_DISTRIBUTION, REPO_TREE
from utils.state import init_session_state


st.set_page_config(
    page_title="Workflow & Data",
    page_icon=":gear:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    init_session_state()
    sidebar_state = render_sidebar()
    inject_theme(sidebar_state["theme_name"])

    render_section_header(
        "Workflow & Data",
        "A technical view of the routed inference pipeline, repository layout, and notebook-derived dataset strategy.",
    )

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.graphviz_chart(
            """
            digraph RoutedPipeline {
                rankdir=LR;
                node [shape=box, style="rounded,filled", color="#7A8CFF", fontcolor="white"];
                Input [label="Input Microscopy Image"];
                Prep [label="Resize + Tensor + Optional Normalize"];
                Broad [label="Broad Classifier\\nEfficientNet-B0"];
                Leuk [label="Leukemia Subtype\\nEfficientNet-B0"];
                Lymph [label="Lymphoma Subtype\\nDenseNet121"];
                Final [label="Final Routed Prediction"];
                Input -> Prep -> Broad;
                Broad -> Leuk [label="LEUKEMIA"];
                Broad -> Lymph [label="LYMPHOMA"];
                Broad -> Final [label="MYELOMA / HEALTHY"];
                Leuk -> Final;
                Lymph -> Final;
            }
            """
        )
    with right:
        st.plotly_chart(
            dataset_distribution_chart(DATASET_DISTRIBUTION),
            use_container_width=True,
        )

    step_cols = st.columns(3, gap="large")
    with step_cols[0]:
        render_note_card(
            "Stage 1: Intake",
            "Images are loaded through Pillow and converted to RGB, including BMP-based myeloma samples.",
        )
    with step_cols[1]:
        render_note_card(
            "Stage 2: Routing",
            "The broad classifier decides whether to terminate early or dispatch to a subtype model.",
        )
    with step_cols[2]:
        render_note_card(
            "Stage 3: Reporting",
            "The frontend surfaces probabilities, charts, session history, and downloadable reports.",
        )

    workflow_tab, files_tab, assumptions_tab = st.tabs(
        ["Pipeline Notes", "Repository Map", "Inferred Assumptions"]
    )

    with workflow_tab:
        st.markdown(
            """
            - The **broad classifier notebook** merges leukemia, lymphoma, myeloma, and healthy images into a single triage task.
            - The **leukemia notebook** filters out healthy-class folders before training and evaluates across 4,000 subtype images.
            - The **lymphoma notebook** uses an 80/20 split with a custom DenseNet121 head and focal loss.
            - The deployment-facing code reconstructs those architectures in `src/load_models.py`.
            """
        )

    with files_tab:
        st.code(REPO_TREE, language="text")

    with assumptions_tab:
        st.markdown(
            """
            - Model checkpoints are treated as authoritative for inference and assumed to match notebook architectures.
            - Reported metrics are extracted from notebook outputs rather than a separate experiment tracker.
            - This frontend assumes local model access and does not yet automate dataset downloads.
            """
        )
        st.caption(APP_COPY["medical_disclaimer"])


if __name__ == "__main__":
    main()
