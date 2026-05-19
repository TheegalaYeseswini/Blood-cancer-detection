from __future__ import annotations

from datetime import datetime

import streamlit as st


DEFAULT_STATE = {
    "theme_name": "Midnight",
    "device_preference": "auto",
    "analyst_name": "Demo Analyst",
    "prediction_history": [],
    "last_result": None,
}


def init_session_state() -> None:
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_prediction_history(result: dict[str, object], source_label: str) -> None:
    broad_prediction = result["tetraclassifier"]
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source_label,
        "broad_label": broad_prediction["predicted_label"],
        "broad_confidence": broad_prediction["confidence"],
        "secondary_label": result["combined"]["secondary_label"],
        "device": result["meta"]["device"],
    }
    st.session_state["prediction_history"] = [entry] + st.session_state["prediction_history"][:9]
    st.session_state["last_result"] = result
