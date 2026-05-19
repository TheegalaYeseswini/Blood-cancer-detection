from __future__ import annotations

from pathlib import Path

import streamlit as st

from components.cards import render_metric_grid, render_note_card, render_section_header
from components.charts import confidence_gauge, history_confidence_chart, probability_bar_chart
from components.theme import inject_theme, render_sidebar
from utils.exports import build_pdf_report, probabilities_to_csv_bytes, result_to_json_bytes
from utils.inference import (
    load_image_from_path,
    probabilities_to_frame,
    read_uploaded_image,
    run_routed_inference,
)
from utils.project_content import APP_COPY, SAMPLE_EXPECTATIONS, get_sample_catalog
from utils.state import add_prediction_history, init_session_state


st.set_page_config(
    page_title="Inference Studio",
    page_icon=":microscope:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    init_session_state()
    sidebar_state = render_sidebar()
    inject_theme(sidebar_state["theme_name"])

    render_section_header(
        "Inference Studio",
        "Upload a microscopy image, capture one from the camera, or test the bundled samples.",
    )

    input_image = None
    source_name = None

    sample_tab, upload_tab, camera_tab = st.tabs(
        ["Sample Gallery", "Upload Image", "Camera Capture"]
    )

    with sample_tab:
        samples = get_sample_catalog()
        sample_lookup = {sample["name"]: sample for sample in samples}
        selected_name = st.selectbox("Choose a bundled sample", list(sample_lookup.keys()))
        selected_sample = sample_lookup[selected_name]
        source_name = selected_name
        input_image = load_image_from_path(selected_sample["path"])
        left, right = st.columns([1, 1.15], gap="large")
        with left:
            st.image(input_image, caption=selected_name, use_container_width=True)
        with right:
            render_note_card("Expected Behavior", SAMPLE_EXPECTATIONS.get(selected_name, "Sample input"))
            st.caption(f"Source path: `{selected_sample['path']}`")

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Upload a blood-cell image",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        )
        if uploaded_file is not None:
            input_image = read_uploaded_image(uploaded_file.getvalue())
            source_name = uploaded_file.name
            st.image(input_image, caption=uploaded_file.name, use_container_width=True)

    with camera_tab:
        camera_file = st.camera_input("Capture a microscopy image")
        if camera_file is not None:
            input_image = read_uploaded_image(camera_file.getvalue())
            source_name = "camera_capture.png"
            st.image(input_image, caption="Camera capture", use_container_width=True)

    run_button = st.button("Run Routed Inference", use_container_width=True, type="primary")
    if run_button and input_image is None:
        st.error("Choose a sample, upload an image, or capture one from the camera first.")

    if run_button and input_image is not None:
        progress_bar = st.progress(0.0)
        status = st.empty()

        def update_progress(progress: float, message: str) -> None:
            progress_bar.progress(progress)
            status.info(message)

        try:
            result = run_routed_inference(
                image=input_image,
                source_name=source_name or "uploaded_image",
                device_preference=sidebar_state["device_preference"],
                progress_callback=update_progress,
            )
            add_prediction_history(result, source_name or "uploaded_image")
            progress_bar.empty()
            status.empty()
            render_results(result, sidebar_state["analyst_name"])
        except Exception as exc:
            progress_bar.empty()
            status.empty()
            st.error(f"Inference failed: {exc}")

    elif st.session_state.get("last_result") is not None:
        render_results(st.session_state["last_result"], sidebar_state["analyst_name"])

    history = st.session_state.get("prediction_history", [])
    if history:
        render_section_header(
            "Session History",
            "Recent predictions are kept in session state for side-by-side review during demos.",
        )
        st.dataframe(history, use_container_width=True, hide_index=True)
        if len(history) > 1:
            st.plotly_chart(history_confidence_chart(history), use_container_width=True)

    st.caption(APP_COPY["medical_disclaimer"])


def render_results(result: dict[str, object], analyst_name: str) -> None:
    broad = result["tetraclassifier"]
    subtype = result["selected_subtype_model"]
    combined = result["combined"]
    meta = result["meta"]

    render_metric_grid(
        [
            {
                "label": "Broad Diagnosis",
                "value": broad["predicted_label"],
                "caption": f"{broad['confidence'] * 100:.2f}% confidence",
            },
            {
                "label": "Subtype Result",
                "value": combined["secondary_label"],
                "caption": combined["used_subtype_model"] or "No subtype model used",
            },
            {
                "label": "Execution Device",
                "value": meta["device"].upper(),
                "caption": "Automatically downgraded to CPU if CUDA is unavailable",
            },
            {
                "label": "Inference Time",
                "value": f"{meta['inference_ms']} ms",
                "caption": f"Analyst label: {analyst_name}",
            },
        ]
    )

    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        st.plotly_chart(
            confidence_gauge(broad["confidence"], "Broad Classifier Confidence"),
            use_container_width=True,
        )
        render_note_card("Routing Summary", combined["summary"])
    with right:
        st.plotly_chart(
            probability_bar_chart(
                broad["probabilities"],
                "Broad Classifier Probabilities",
            ),
            use_container_width=True,
        )

    if subtype is not None:
        render_section_header(
            "Subtype Follow-up",
            "Because the broad prediction triggered routing, a second model was run to refine the diagnosis.",
        )
        st.plotly_chart(
            probability_bar_chart(
                subtype["probabilities"],
                f"{subtype['model_name']} Probabilities",
            ),
            use_container_width=True,
        )

    probabilities_frame = probabilities_to_frame(result)
    export_cols = st.columns(3, gap="medium")
    with export_cols[0]:
        st.download_button(
            "Download JSON",
            data=result_to_json_bytes(result),
            file_name="prediction_result.json",
            mime="application/json",
            use_container_width=True,
        )
    with export_cols[1]:
        st.download_button(
            "Download CSV",
            data=probabilities_to_csv_bytes(probabilities_frame),
            file_name="prediction_probabilities.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with export_cols[2]:
        st.download_button(
            "Download PDF Report",
            data=build_pdf_report(result, analyst_name),
            file_name="prediction_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with st.expander("Raw Result Payload"):
        st.json(result)


if __name__ == "__main__":
    main()
