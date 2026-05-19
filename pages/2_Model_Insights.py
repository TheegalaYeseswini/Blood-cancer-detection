from __future__ import annotations

import pandas as pd
import streamlit as st

from components.cards import render_metric_grid, render_note_card, render_section_header
from components.charts import class_metric_heatmap, dataset_distribution_chart, model_accuracy_chart
from components.theme import inject_theme, render_sidebar
from utils.project_content import (
    BROAD_REPORT,
    DATASET_DISTRIBUTION,
    LEUKEMIA_REPORT,
    LYMPHOMA_REPORT,
    MODEL_BENCHMARKS,
    MODEL_CATALOG,
    NOTEBOOK_SUMMARIES,
)
from utils.state import init_session_state


st.set_page_config(
    page_title="Model Insights",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    init_session_state()
    sidebar_state = render_sidebar()
    inject_theme(sidebar_state["theme_name"])

    render_section_header(
        "Model Insights",
        "Architecture choices, notebook metrics, and training design decisions extracted from the repository.",
    )

    render_metric_grid(
        [
            {
                "label": "Broad Model",
                "value": "EfficientNet-B0",
                "caption": "4-class routed entry point",
            },
            {
                "label": "Leukemia Model",
                "value": "EfficientNet-B0",
                "caption": "4-class subtype head",
            },
            {
                "label": "Lymphoma Model",
                "value": "DenseNet121",
                "caption": "3-class subtype head",
            },
            {
                "label": "Routing Rule",
                "value": "Hierarchical",
                "caption": "Subtype only runs after matching broad class",
            },
        ]
    )

    perf_tab, architecture_tab, notebook_tab = st.tabs(
        ["Performance", "Model Catalog", "Notebook Lineage"]
    )

    with perf_tab:
        top_left, top_right = st.columns([1, 1], gap="large")
        with top_left:
            st.plotly_chart(model_accuracy_chart(MODEL_BENCHMARKS), use_container_width=True)
        with top_right:
            st.plotly_chart(
                dataset_distribution_chart(DATASET_DISTRIBUTION),
                use_container_width=True,
            )

        heatmap_cols = st.columns(3, gap="large")
        with heatmap_cols[0]:
            st.plotly_chart(
                class_metric_heatmap(BROAD_REPORT, "Broad Classifier Report"),
                use_container_width=True,
            )
        with heatmap_cols[1]:
            st.plotly_chart(
                class_metric_heatmap(LEUKEMIA_REPORT, "Leukemia Subtype Report"),
                use_container_width=True,
            )
        with heatmap_cols[2]:
            st.plotly_chart(
                class_metric_heatmap(LYMPHOMA_REPORT, "Lymphoma Subtype Report"),
                use_container_width=True,
            )

    with architecture_tab:
        for model in MODEL_CATALOG:
            render_note_card(model["name"], f"{model['backbone']} | {model['metric']}")
            st.markdown(
                f"""
                - **Checkpoint:** `{model['checkpoint']}`
                - **Labels:** `{", ".join(model['labels'])}`
                - **Usage:** {model['usage']}
                """
            )

        st.dataframe(pd.DataFrame(MODEL_CATALOG), use_container_width=True, hide_index=True)

    with notebook_tab:
        for notebook in NOTEBOOK_SUMMARIES:
            with st.expander(notebook["notebook"], expanded=False):
                st.markdown(f"**Task:** {notebook['task']}")
                st.markdown(f"**Backbone:** {notebook['backbone']}")
                st.markdown("**Hyperparameters**")
                for item in notebook["hyperparameters"]:
                    st.markdown(f"- {item}")
                st.markdown("**Augmentations**")
                for item in notebook["augmentations"]:
                    st.markdown(f"- {item}")


if __name__ == "__main__":
    main()
