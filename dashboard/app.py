"""
Dashboard Entry Point
Renders the Streamlit UI for prediction, metrics, error analysis, and calibration review.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# Add src/ to import path so dashboard can reuse pipeline modules when launched from dashboard/.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import dashboard helpers after path setup; noqa keeps this intentional import order lint-clean.
from utils import (  
    append_prediction_row,
    discover_scopes,
    get_classification_reports,
    get_common_confusion_images,
    get_confusion_images,
    get_dashboard_artifact_paths,
    list_any_misclassified_images,
    list_misclassified_images,
    list_model_files,
    path_last_updated,
    read_csv_safe,
    resolve_default_model,
    run_uploaded_prediction,
)


# Configure page metadata before rendering any Streamlit elements.
st.set_page_config(
    page_title="Recyclyst Dashboard",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _inject_styles() -> None:
    # Inject one global CSS theme so all pages share the same visual language.
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Manrope:wght@400;600;700&display=swap');

        :root {
            --bg-0: #f4f8f7;
            --bg-1: #e7f2ef;
            --text-main: #12232e;
            --text-soft: #355564;
            --accent-1: #0f766e;
            --accent-2: #f97316;
            --card: rgba(255, 255, 255, 0.86);
            --stroke: rgba(18, 35, 46, 0.12);
        }

        .stApp {
            font-family: 'Manrope', sans-serif;
            color: var(--text-main);
            background:
                radial-gradient(1200px 500px at -10% -10%, #d5f0ea 0%, transparent 60%),
                radial-gradient(900px 400px at 120% 10%, #ffe2c8 0%, transparent 55%),
                linear-gradient(180deg, var(--bg-0) 0%, #f9fbfb 100%);
        }

        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        #MainMenu,
        footer {
            display: none !important;
        }

        .block-container {
            padding-top: 0.25rem;
        }

        .hero {
            border: 1px solid var(--stroke);
            border-radius: 18px;
            padding: 22px 24px;
            background: linear-gradient(120deg, rgba(15,118,110,0.14) 0%, rgba(249,115,22,0.13) 100%);
            box-shadow: 0 10px 30px rgba(9, 26, 33, 0.08);
            margin-bottom: 12px;
        }

        .hero h1 {
            margin: 0;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2rem;
            letter-spacing: -0.02em;
        }

        .hero p {
            margin: 8px 0 0 0;
            color: var(--text-soft);
            font-size: 0.98rem;
        }

        .top-nav-brand-wrap {
            height: 42px;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            padding-left: 6px;
        }

        .top-nav-brand {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.02rem;
            font-weight: 600;
            color: #dbe7ff;
            line-height: 1.2;
            white-space: nowrap;
        }

        [data-testid="stHorizontalBlock"]:has(.top-nav-brand) {
            background: linear-gradient(120deg, #0f1628, #171d33);
            border: 1px solid rgba(73, 88, 121, 0.5);
            border-radius: 14px;
            padding: 8px 12px;
            margin-bottom: 12px;
            box-shadow: 0 8px 26px rgba(7, 13, 28, 0.25);
            align-items: center !important;
        }

        .card {
            border: 1px solid var(--stroke);
            border-radius: 14px;
            padding: 14px 16px;
            background: var(--card);
            backdrop-filter: blur(6px);
            box-shadow: 0 8px 22px rgba(9, 26, 33, 0.05);
        }

        .kpi-title {
            color: var(--text-soft);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 3px;
        }

        .kpi-value {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 1.3rem;
            color: var(--text-main);
        }

        .small-note {
            color: var(--text-soft);
            font-size: 0.86rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(7, 89, 82, 0.88), rgba(21, 52, 68, 0.93));
        }

        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] [data-baseweb="select"] * {
            color: #ebfffa !important;
        }

        [data-testid="stSidebar"] .material-icons,
        [data-testid="stSidebar"] [class*="material-symbols"] {
            font-family: 'Material Icons', 'Material Symbols Rounded' !important;
            font-feature-settings: 'liga' !important;
        }

        .stButton > button,
        .stDownloadButton > button {
            border: none !important;
            border-radius: 12px !important;
            background: linear-gradient(120deg, #0f766e, #0ea5a2) !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            transition: all 0.2s ease-in-out !important;
            box-shadow: 0 8px 20px rgba(15, 118, 110, 0.25) !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 12px 26px rgba(15, 118, 110, 0.30) !important;
        }

        .stButton > button:disabled,
        .stDownloadButton > button:disabled {
            background: #cfd8dc !important;
            color: #5f717a !important;
            box-shadow: none !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            border: 1px solid rgba(148, 163, 184, 0.45) !important;
            background: linear-gradient(120deg, #1f2433, #262939) !important;
        }

        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzoneInstructions"] span {
            color: #d9e8f5 !important;
        }

        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {
            background: linear-gradient(120deg, #0f766e, #0ea5a2) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.28) !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
        }

        [data-testid="stFileUploaderDropzone"] button:hover,
        [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"]:hover {
            filter: brightness(1.07);
        }

        [data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--stroke);
        }

        .stTabs [role="tab"] {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
        }

        [data-testid="stHorizontalBlock"]:has(.top-nav-brand) [data-baseweb="radio"] {
            width: 100%;
        }

        [data-testid="stHorizontalBlock"]:has(.top-nav-brand) [role="radiogroup"] {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            align-items: center !important;
            justify-content: center;
            gap: 7px;
            width: fit-content;
            margin: 1px auto 0 auto;
        }

        [data-testid="stHorizontalBlock"]:has(.top-nav-brand) [role="radiogroup"] > label {
            display: flex !important;
            align-items: center !important;
            border-radius: 10px;
            border: 1px solid rgba(122, 136, 173, 0.35);
            background: rgba(34, 43, 68, 0.72);
            color: #cfd8f6 !important;
            padding: 8px 13px;
            margin: 0;
            min-height: 39px;
            white-space: nowrap;
        }

        [data-testid="stHorizontalBlock"]:has(.top-nav-brand) [role="radiogroup"] > label > div:first-child {
            display: none !important;
        }

        [data-testid="stHorizontalBlock"]:has(.top-nav-brand) [role="radiogroup"] > label input[type="radio"] {
            display: none !important;
        }

        [data-testid="stHorizontalBlock"]:has(.top-nav-brand) [role="radiogroup"] > label:has(input:checked) {
            background: linear-gradient(120deg, #2563eb, #1d4ed8);
            color: #ffffff !important;
            border-color: rgba(170, 201, 255, 0.75);
            box-shadow: 0 6px 18px rgba(37, 99, 235, 0.25);
        }

        [data-testid="stHorizontalBlock"]:has(.top-nav-brand) [role="radiogroup"] > label p {
            color: inherit !important;
            font-weight: 600;
            font-size: 0.9rem;
            margin: 0 !important;
        }

        [data-testid="stHorizontalBlock"]:has(.top-nav-brand) [data-testid="stWidgetLabel"] p {
            color: #dbe7ff !important;
            font-weight: 600;
            font-size: 0.88rem;
        }

        [data-testid="stHorizontalBlock"]:has(.top-nav-brand) [data-baseweb="select"] > div {
            background: rgba(34, 43, 68, 0.92);
            border: 1px solid rgba(104, 118, 154, 0.55);
            min-height: 36px;
        }

        [data-testid="stHorizontalBlock"]:has(.top-nav-brand) [data-baseweb="select"] * {
            color: #ecf2ff !important;
        }

        [data-testid="stHorizontalBlock"]:has(.top-nav-brand) [data-baseweb="select"] svg {
            fill: #ecf2ff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header(scope_label: str, show_title: bool = False) -> None:
    # Keep the full dashboard title only on Overview while reusing subtitle text everywhere.
    title_html = "<h1>Machine Learning Dashboard</h1>" if show_title else ""
    subtitle_html = (
        f"<p>Interactive workspace for <strong>{scope_label}</strong>. "
        "Review metrics, run predictions, inspect errors, and check calibration artifacts.</p>"
    )
    header_html = f"<div class=\"hero\">{title_html}{subtitle_html}</div>"
    st.markdown(header_html, unsafe_allow_html=True)


def _render_artifact_status(scope) -> None:
    # Render quick availability cards for key files in the active workspace scope.
    artifact_paths = get_dashboard_artifact_paths(scope)

    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]
    for index, (name, path) in enumerate(artifact_paths.items()):
        exists = "Available" if path.exists() else "Missing"
        updated = path_last_updated(path)
        with columns[index % 3]:
            st.markdown(
                f"""
                <div class="card">
                    <div class="kpi-title">{name}</div>
                    <div class="kpi-value">{exists}</div>
                    <div class="small-note">Updated: {updated}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_highlights(scope, table_rows: int) -> None:
    # Load shared metrics/plot artifacts used by the overview highlight cards and charts.
    st.markdown("### Performance Highlights")
    metrics_global = scope.results_dir / "metrics" / "global"
    plots_global = scope.results_dir / "plots" / "global"

    evaluation_summary = read_csv_safe(metrics_global / "evaluation_summary.csv")
    top_confusions = read_csv_safe(metrics_global / "top_confusions.csv")
    worst_classes = read_csv_safe(metrics_global / "worst_classes.csv")
    calibration_metrics = read_csv_safe(metrics_global / "calibration_metrics.csv")
    reliability_bins = read_csv_safe(metrics_global / "reliability_bins.csv")
    reliability_curve = plots_global / "reliability_curve.png"

    # Split overview into performance summary (left) and error snapshot (right).
    left, right = st.columns([1.35, 1])

    with left:
        st.markdown("#### Accuracy & Macro-F1 by model")
        if evaluation_summary is not None and not evaluation_summary.empty:
            st.dataframe(evaluation_summary.head(table_rows), use_container_width=True)

            if {"model_name", "accuracy", "macro_f1"}.issubset(evaluation_summary.columns):
                metrics_long = evaluation_summary[["model_name", "accuracy", "macro_f1"]].melt(
                    id_vars=["model_name"],
                    value_vars=["accuracy", "macro_f1"],
                    var_name="metric",
                    value_name="score",
                )
                bar = px.bar(
                    metrics_long,
                    x="model_name",
                    y="score",
                    color="metric",
                    barmode="group",
                    text="score",
                    color_discrete_sequence=["#0f766e", "#f97316"],
                    title="Model quality overview",
                )
                bar.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                bar.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10), yaxis_range=[0, 1])
                st.plotly_chart(bar, use_container_width=True)
        else:
            st.info("Run evaluation to populate the highlight section.")

    with right:
        st.markdown("#### Error pattern snapshot")
        if top_confusions is not None and not top_confusions.empty and {"true_label", "pred_label", "count"}.issubset(top_confusions.columns):
            view_df = top_confusions.head(min(table_rows, 10)).copy()
            view_df["pair"] = view_df["true_label"].astype(str) + " -> " + view_df["pred_label"].astype(str)
            conf_chart = px.bar(
                view_df,
                x="count",
                y="pair",
                orientation="h",
                color="count",
                color_continuous_scale="Sunsetdark",
                title="Top confusion pairs",
            )
            conf_chart.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10), coloraxis_showscale=False)
            st.plotly_chart(conf_chart, use_container_width=True)
        else:
            st.info("Run error analysis to display confusion highlights.")

        if worst_classes is not None and not worst_classes.empty:
            st.dataframe(worst_classes.head(table_rows), use_container_width=True)

    # Surface calibration quality directly on overview to avoid extra page switches.
    st.markdown("#### Calibration at a glance")
    c1, c2 = st.columns([1, 1])
    with c1:
        if calibration_metrics is not None and not calibration_metrics.empty:
            st.dataframe(calibration_metrics, use_container_width=True)
        else:
            st.info("Calibration metrics are missing. Use the sidebar action to generate them.")

        if reliability_bins is not None and not reliability_bins.empty:
            plot_df = reliability_bins[reliability_bins.get("count", 0) > 0].copy() if "count" in reliability_bins.columns else reliability_bins.copy()
            if {"avg_confidence", "accuracy"}.issubset(plot_df.columns) and not plot_df.empty:
                rel_chart = px.line(
                    plot_df,
                    x="avg_confidence",
                    y="accuracy",
                    markers=True,
                    title="Reliability trend",
                    color_discrete_sequence=["#0f766e"],
                )
                rel_chart.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash", color="#64748b"))
                rel_chart.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), xaxis_range=[0, 1], yaxis_range=[0, 1])
                st.plotly_chart(rel_chart, use_container_width=True)

    with c2:
        if reliability_curve.exists():
            st.image(str(reliability_curve), caption="Reliability curve", use_container_width=True)
        else:
            st.info("Reliability curve is missing. Generate calibration from the sidebar.")


def _render_predict_tab(scope) -> None:
    # Prediction tab reuses trained checkpoints and the existing src/predict inference flow.
    st.subheader("Live Prediction")
    st.caption("Upload an image, select a model, and run top-3 class inference.")

    # Resolve model list and pick the same default the pipeline marks as current best.
    model_files = list_model_files(scope)
    default_model = resolve_default_model(scope)

    if not model_files:
        st.warning(f"No .keras model files found in `{scope.models_dir}`.")
        return

    model_labels = [str(path.relative_to(PROJECT_ROOT)) for path in model_files]
    default_index = 0
    if default_model in model_files:
        default_index = model_files.index(default_model)

    selected_label = st.selectbox("Model checkpoint", model_labels, index=default_index)
    selected_model = PROJECT_ROOT / selected_label

    # Accept common image formats from users for one-click inference.
    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="The dashboard uses your existing `src/predict.py` pipeline for inference.",
    )

    save_prediction = st.checkbox("Save this prediction in metrics log", value=True)
    log_path = scope.results_dir / "metrics" / "global" / "prediction_examples_dashboard.csv"

    # Run inference only after both file upload and explicit user trigger.
    if uploaded and st.button("Run Prediction", type="primary"):
        with st.spinner("Running model inference..."):
            result = run_uploaded_prediction(
                image_bytes=uploaded.getvalue(),
                model_path=selected_model,
                class_names_path=scope.class_names_path,
            )

        c1, c2, c3 = st.columns(3)
        c1.metric("Top prediction", str(result.get("top_prediction", "-")))
        c2.metric("Confidence", f"{float(result.get('confidence_percent', 0.0)):.2f}%")
        c3.metric("Model", str(result.get("model_name", "-")))

        top3 = pd.DataFrame(result.get("top_3", []))
        if not top3.empty:
            top3 = top3.rename(columns={"label": "Class", "confidence": "Confidence (%)"})
            chart = px.bar(
                top3,
                x="Confidence (%)",
                y="Class",
                orientation="h",
                text="Confidence (%)",
                color="Confidence (%)",
                color_continuous_scale="Tealgrn",
                title="Top-3 prediction confidence",
            )
            chart.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10), coloraxis_showscale=False)
            chart.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            st.plotly_chart(chart, use_container_width=True)

        # Optionally append results for traceability and quick demo logs.
        if save_prediction:
            append_prediction_row(log_path, uploaded.name, result)
            st.success(f"Prediction row appended to `{log_path.relative_to(PROJECT_ROOT)}`")


def _render_metrics_tab(scope) -> None:
    # Metrics tab combines global comparisons with per-model classification reports.
    st.subheader("Training & Evaluation")

    metrics_global = scope.results_dir / "metrics" / "global"
    model_comparison = read_csv_safe(metrics_global / "model_comparison.csv")
    evaluation_summary = read_csv_safe(metrics_global / "evaluation_summary.csv")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Model Comparison")
        if model_comparison is not None and not model_comparison.empty:
            st.dataframe(model_comparison, use_container_width=True)
        else:
            st.info("`model_comparison.csv` not available yet.")

    with col_b:
        st.markdown("#### Evaluation Summary")
        if evaluation_summary is not None and not evaluation_summary.empty:
            st.dataframe(evaluation_summary, use_container_width=True)
        else:
            st.info("`evaluation_summary.csv` not available yet.")

    if evaluation_summary is not None and not evaluation_summary.empty:
        if {"model_name", "accuracy", "macro_f1"}.issubset(evaluation_summary.columns):
            scatter = px.scatter(
                evaluation_summary,
                x="accuracy",
                y="macro_f1",
                text="model_name",
                size_max=18,
                title="Accuracy vs Macro-F1",
                color="macro_f1",
                color_continuous_scale="Aggrnyl",
            )
            scatter.update_traces(textposition="top center")
            scatter.update_layout(height=390, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(scatter, use_container_width=True)

    # Report selector lets users inspect class-wise precision/recall for each model.
    st.markdown("#### Classification Reports")
    reports = get_classification_reports(scope)
    if not reports:
        st.info("No per-model `classification_report.csv` files found.")
        return

    selected_model = st.selectbox("Select model report", list(reports.keys()))
    report_df = read_csv_safe(reports[selected_model])
    if report_df is None or report_df.empty:
        st.warning("Could not load selected classification report.")
    else:
        st.dataframe(report_df, use_container_width=True)


def _render_error_tab(scope, gallery_limit: int = 12) -> None:
    # Error tab summarizes where predictions fail and visualizes failure examples.
    st.subheader("Error Analysis")

    metrics_global = scope.results_dir / "metrics" / "global"
    top_confusions = read_csv_safe(metrics_global / "top_confusions.csv")
    worst_classes = read_csv_safe(metrics_global / "worst_classes.csv")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Top Confusions")
        if top_confusions is not None and not top_confusions.empty:
            st.dataframe(top_confusions, use_container_width=True)
        else:
            st.info("`top_confusions.csv` not available.")

    with c2:
        st.markdown("#### Worst Recall Classes")
        if worst_classes is not None and not worst_classes.empty:
            st.dataframe(worst_classes, use_container_width=True)
        else:
            st.info("`worst_classes.csv` not available.")

    # Show confusion artifacts first, with graceful fallbacks when files are missing.
    st.markdown("#### Confusion Visuals")
    confusion_images = get_confusion_images(scope)
    common_images = get_common_confusion_images(scope)

    if not confusion_images:
        st.info("No confusion matrix images found in plots.")
        return

    model_choice = st.selectbox("Model confusion view", list(confusion_images.keys()))

    v1, v2 = st.columns(2)
    with v1:
        st.image(str(confusion_images[model_choice]), caption=f"{model_choice} confusion matrix", use_container_width=True)
    with v2:
        if model_choice in common_images:
            st.image(str(common_images[model_choice]), caption=f"{model_choice} common confusions", use_container_width=True)
        elif common_images:
            fallback_model, fallback_path = next(iter(common_images.items()))
            st.image(str(fallback_path), caption=f"{fallback_model} common confusions (fallback view)", use_container_width=True)
        elif top_confusions is not None and not top_confusions.empty and {"true_label", "pred_label", "count"}.issubset(top_confusions.columns):
            top_view = top_confusions.head(10).copy()
            top_view["pair"] = top_view["true_label"].astype(str) + " -> " + top_view["pred_label"].astype(str)
            fallback_chart = px.bar(
                top_view,
                x="count",
                y="pair",
                orientation="h",
                color="count",
                color_continuous_scale="Sunsetdark",
                title="Top confusion pairs",
            )
            fallback_chart.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), coloraxis_showscale=False)
            st.plotly_chart(fallback_chart, use_container_width=True)
        else:
            st.info("No `common_confusions.png` found for this model.")

    # Build a misclassified image gallery, falling back to any available model samples.
    st.markdown("#### Misclassified Example Gallery")
    sample_paths = list_misclassified_images(scope, model_choice, limit=gallery_limit)
    showing_fallback = False
    if not sample_paths:
        sample_paths = list_any_misclassified_images(scope, limit=gallery_limit)
        showing_fallback = bool(sample_paths)
        if sample_paths:
            st.caption("Showing fallback gallery from other available models.")
        else:
            st.info("No misclassified images found yet in this scope.")
            return

    # Filter extreme aspect ratios so gallery tiles stay visually consistent.
    display_paths = []
    for image_path in sample_paths:
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception:
            continue

        if width <= 0 or height <= 0:
            continue

        aspect_ratio = width / height
        if aspect_ratio < 0.58 or aspect_ratio > 1.85:
            continue

        display_paths.append(image_path)

    if not display_paths:
        display_paths = sample_paths

    if showing_fallback:
        st.caption(f"Displaying {len(display_paths)} recent samples.")

    cols = st.columns(4)
    for index, image_path in enumerate(display_paths):
        try:
            with Image.open(image_path) as img:
                thumb = ImageOps.fit(img.convert("RGB"), (360, 360), centering=(0.5, 0.5))
            cols[index % 4].image(thumb, caption=image_path.name[:52], use_container_width=True)
        except Exception:
            cols[index % 4].image(str(image_path), caption=image_path.name[:52], use_container_width=True)


def _render_calibration_tab(scope) -> None:
    # Calibration tab checks confidence alignment against true accuracy.
    st.subheader("Calibration")

    metrics_global = scope.results_dir / "metrics" / "global"
    plots_global = scope.results_dir / "plots" / "global"

    calibration_metrics = read_csv_safe(metrics_global / "calibration_metrics.csv")
    reliability_bins = read_csv_safe(metrics_global / "reliability_bins.csv")
    reliability_curve = plots_global / "reliability_curve.png"

    st.markdown("#### Calibration Metrics")
    if calibration_metrics is not None and not calibration_metrics.empty:
        st.dataframe(calibration_metrics, use_container_width=True)
    else:
        st.info("`calibration_metrics.csv` not available.")

    # Place trend chart and saved curve image side-by-side for faster comparison.
    left, right = st.columns(2)
    with left:
        st.markdown("#### Reliability Trend")
        if reliability_bins is None or reliability_bins.empty:
            st.info("`reliability_bins.csv` not available.")
        else:
            plot_df = reliability_bins[reliability_bins["count"] > 0].copy() if "count" in reliability_bins.columns else reliability_bins.copy()
            if {"avg_confidence", "accuracy"}.issubset(plot_df.columns) and not plot_df.empty:
                curve = px.line(
                    plot_df,
                    x="avg_confidence",
                    y="accuracy",
                    markers=True,
                    title="Calibration trend (accuracy vs confidence)",
                )
                curve.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash", color="#6b7280"))
                curve.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), xaxis_range=[0, 1], yaxis_range=[0, 1])
                st.plotly_chart(curve, use_container_width=True)
            else:
                st.info("Reliability bins are present but missing plotting columns.")

    with right:
        st.markdown("#### Reliability Curve")
        if reliability_curve.exists():
            st.image(str(reliability_curve), caption="Reliability curve", use_container_width=True)
        else:
            st.info("`reliability_curve.png` not found.")

    st.markdown("#### Reliability Bins")
    if reliability_bins is not None and not reliability_bins.empty:
        st.dataframe(reliability_bins, use_container_width=True)
    else:
        st.info("`reliability_bins.csv` not available.")


def _render_artifacts_tab(scope) -> None:
    # Artifact browser gives a quick audit table for expected output files.
    st.subheader("Artifact Browser")
    st.caption("Use this to confirm what is current after every new training/evaluation run.")

    artifact_paths = get_dashboard_artifact_paths(scope)
    rows = [
        {
            "Artifact": name,
            "Path": str(path.relative_to(PROJECT_ROOT)),
            "Status": "Available" if path.exists() else "Missing",
            "Last updated": path_last_updated(path),
        }
        for name, path in artifact_paths.items()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _render_help_tab(scope) -> None:
    # Help page documents project purpose, workflow, and what each dashboard page covers.
    st.subheader("Help & Project Guide")
    st.markdown(
        """
        ### Project purpose
        This project classifies recyclable waste images into material classes using transfer learning. The goal is to provide reliable model performance, clear evaluation artifacts, and an accessible interface for prediction and review.

        ### How the system works
        1. Build deterministic train/validation/test splits.
        2. Run EDA to generate dataset summary artifacts.
        3. Train baseline and transfer-learning models.
        4. Evaluate model quality on the test set.
        5. Run error analysis and calibration for reliability insights.
        6. Use this dashboard for live prediction and visual artifact review.

        ### Models used in the stable pipeline
        - **baseline_cnn**: lightweight custom CNN baseline used as a reference point.
        - **resnet50_frozen**: ResNet50 feature extractor with frozen backbone layers.
        - **resnet50_finetuned**: ResNet50 with selective backbone unfreezing for improved adaptation (stable best model).

        ### What each dashboard page provides
        - **Overview**: top-level performance and calibration highlights.
        - **Prediction**: upload an image and get top-3 class predictions.
        - **Metrics**: model comparison, evaluation summaries, classification reports.
        - **Error Analysis**: confusion artifacts and misclassified sample gallery.
        - **Calibration**: reliability metrics, bins, and calibration trend/curve.
        - **Artifacts**: status table of key files currently available in scope.
        """
    )

    st.markdown("### Active scope")
    st.code(str(scope.results_dir.relative_to(PROJECT_ROOT)), language="text")


def main() -> None:
    # Apply UI styles once and configure shared display limits.
    _inject_styles()
    table_rows = 10
    gallery_limit = 12

    # Discover all available result scopes and map labels for the top-right selector.
    scopes = discover_scopes()
    scope_labels = {scope.label: scope for scope in scopes}

    # Top navigation layout: brand (left), pages (center), workspace picker (right).
    top_left, top_mid, top_right = st.columns([1.9, 5.2, 1.9])
    with top_left:
        st.markdown(
            '<div class="top-nav-brand-wrap"><span class="top-nav-brand">Recyclyst Dashboard</span></div>',
            unsafe_allow_html=True,
        )
    with top_mid:
        page = st.radio(
            "Main Navigation",
            ["Overview", "Prediction", "Metrics", "Error Analysis", "Calibration", "Artifacts", "Help"],
            horizontal=True,
            label_visibility="collapsed",
            key="top_navigation",
        )
    with top_right:
        selected_scope_label = st.selectbox("Workspace", list(scope_labels.keys()), key="workspace_top")
        selected_scope = scope_labels[selected_scope_label]

    # Keep the hero subtitle scope-aware and only show title text on the Overview page.
    _render_header(selected_scope.label, show_title=(page == "Overview"))

    # Route the selected top navigation page to its corresponding renderer.
    if page == "Overview":
        _render_highlights(selected_scope, table_rows=table_rows)
    elif page == "Prediction":
        _render_predict_tab(selected_scope)
    elif page == "Metrics":
        _render_metrics_tab(selected_scope)
    elif page == "Error Analysis":
        _render_error_tab(selected_scope, gallery_limit=gallery_limit)
    elif page == "Calibration":
        _render_calibration_tab(selected_scope)
    elif page == "Artifacts":
        _render_artifacts_tab(selected_scope)
    else:
        _render_help_tab(selected_scope)


if __name__ == "__main__":
    main()
