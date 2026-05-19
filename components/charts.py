from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _base_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E5ECF4"},
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return fig


def probability_bar_chart(probabilities: dict[str, float], title: str) -> go.Figure:
    frame = pd.DataFrame(
        {
            "Label": list(probabilities.keys()),
            "Probability": [value * 100 for value in probabilities.values()],
        }
    ).sort_values("Probability", ascending=True)
    fig = px.bar(
        frame,
        x="Probability",
        y="Label",
        orientation="h",
        color="Probability",
        color_continuous_scale=["#1B9AAA", "#6C8CFF", "#A678F0"],
        title=title,
    )
    fig.update_traces(texttemplate="%{x:.2f}%", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, xaxis_title="Confidence (%)", yaxis_title="")
    return _base_layout(fig)


def confidence_gauge(confidence: float, title: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={"suffix": "%", "font": {"size": 30}},
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#7A8CFF"},
                "steps": [
                    {"range": [0, 50], "color": "rgba(122, 140, 255, 0.2)"},
                    {"range": [50, 80], "color": "rgba(27, 154, 170, 0.25)"},
                    {"range": [80, 100], "color": "rgba(54, 211, 153, 0.25)"},
                ],
            },
        )
    )
    return _base_layout(fig)


def model_accuracy_chart(benchmarks: list[dict[str, object]]) -> go.Figure:
    frame = pd.DataFrame(benchmarks)
    fig = px.bar(
        frame,
        x="model",
        y="score",
        color="metric_type",
        text="score_label",
        title="Notebook Benchmark Snapshot",
        barmode="group",
        color_discrete_sequence=["#7A8CFF", "#36D399", "#1B9AAA"],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="", yaxis_title="Score (%)", legend_title="")
    return _base_layout(fig)


def dataset_distribution_chart(dataset_distribution: list[dict[str, object]]) -> go.Figure:
    frame = pd.DataFrame(dataset_distribution)
    fig = px.bar(
        frame,
        x="class_group",
        y="images",
        color="class_group",
        title="Broad Classifier Training Pool",
        color_discrete_sequence=["#7A8CFF", "#A678F0", "#1B9AAA", "#36D399"],
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Image Count")
    fig.update_traces(texttemplate="%{y}", textposition="outside")
    return _base_layout(fig)


def class_metric_heatmap(class_metrics: list[dict[str, object]], title: str) -> go.Figure:
    frame = pd.DataFrame(class_metrics)
    heatmap_frame = frame.set_index("label")[["precision", "recall", "f1"]]
    fig = px.imshow(
        heatmap_frame,
        text_auto=".2f",
        color_continuous_scale=["#0F172A", "#1B9AAA", "#7A8CFF"],
        aspect="auto",
        title=title,
    )
    fig.update_layout(coloraxis_colorbar_title="Score")
    return _base_layout(fig)


def history_confidence_chart(history: list[dict[str, object]]) -> go.Figure:
    frame = pd.DataFrame(history)
    frame["broad_confidence_pct"] = frame["broad_confidence"] * 100
    fig = px.line(
        frame,
        x="timestamp",
        y="broad_confidence_pct",
        markers=True,
        title="Prediction Confidence History",
        color_discrete_sequence=["#7A8CFF"],
    )
    fig.update_layout(xaxis_title="Timestamp", yaxis_title="Broad Confidence (%)")
    return _base_layout(fig)
