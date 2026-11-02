"""MediSynth dashboard for the completed heart-disease ML project.

Run from the project root with: streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import HeartDiseasePredictor


DASHBOARD_DATA = ROOT / "results" / "dashboard"
NAV_ITEMS = ["Home", "Predictions", "Model Evaluation", "Confusion Matrix", "Reports"]
COLORS = {
    "page": "#F5F7FA",
    "card": "#FFFFFF",
    "blue": "#2563EB",
    "blue_light": "#3B82F6",
    "teal": "#0D9488",
    "text": "#1E293B",
    "muted": "#64748B",
    "border": "#E2E8F0",
    "pale_blue": "#EFF6FF",
    "pale_teal": "#F0FDFA",
}
MODEL_COLORS = {
    "Logistic Regression": COLORS["blue_light"],
    "Random Forest": COLORS["teal"],
}
REPORT_FILES = [
    "model_performance.csv",
    "predictions.csv",
    "confusion_matrix.csv",
    "roc_curve.csv",
    "dataset_summary.csv",
]


st.set_page_config(
    page_title="MediSynth | Heart Disease Prediction",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data(show_spinner="Loading cleaned dataset…")
def load_cleaned_dataset() -> pd.DataFrame:
    columns = ["age_years", "ap_hi", "ap_lo", "cholesterol", "gluc", "cardio"]
    path = ROOT / "data" / "processed" / "cardio_cleaned"
    return pd.read_parquet(path, columns=columns)


@st.cache_data(show_spinner=False)
def load_dashboard_csv(filename: str) -> pd.DataFrame:
    if filename not in REPORT_FILES:
        raise ValueError(f"Unknown dashboard dataset: {filename}")
    path = DASHBOARD_DATA / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required Phase 5 dashboard dataset is missing: {path}")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_report_bytes(filename: str) -> bytes:
    if filename not in REPORT_FILES:
        raise ValueError(f"Unknown dashboard report: {filename}")
    path = DASHBOARD_DATA / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required Phase 5 dashboard report is missing: {path}")
    return path.read_bytes()


@st.cache_resource(show_spinner="Loading saved Random Forest and fitted preprocessing pipeline…")
def load_predictor() -> HeartDiseasePredictor:
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName("MediSynthDashboardInference")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return HeartDiseasePredictor(spark)


def load_css() -> None:
    stylesheet = Path(__file__).with_name("styles.css")
    if not stylesheet.is_file():
        raise FileNotFoundError(f"Dashboard stylesheet is missing: {stylesheet}")
    st.markdown(
        f"<style>{stylesheet.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


def make_chart(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_traces(hoverinfo="skip")
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font={"family": "Inter, Segoe UI, sans-serif", "color": COLORS["text"], "size": 12},
        margin={"l": 16, "r": 16, "t": 24, "b": 16},
        hovermode=False,
        dragmode=False,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"color": COLORS["muted"], "size": 11},
        },
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=COLORS["border"],
        tickfont={"color": COLORS["muted"]},
        title_font={"color": COLORS["muted"]},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLORS["border"],
        zeroline=False,
        linecolor=COLORS["border"],
        tickfont={"color": COLORS["muted"]},
        title_font={"color": COLORS["muted"]},
    )
    return fig


def show_chart(fig: go.Figure, *, key: str, height: int = 340) -> None:
    st.plotly_chart(
        make_chart(fig, height),
        use_container_width=True,
        key=key,
        config={
            "staticPlot": True,
            "displayModeBar": False,
            "scrollZoom": False,
            "doubleClick": False,
            "showTips": False,
            "responsive": True,
        },
    )


def section_heading(title: str, caption: str | None = None) -> None:
    st.markdown(f'<h2 class="section-title">{title}</h2>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<p class="section-caption">{caption}</p>', unsafe_allow_html=True)


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.markdown('<div class="eyebrow">HEART DISEASE PREDICTION · PYSPARK ML</div>', unsafe_allow_html=True)
    st.markdown(f'<h1 class="page-title">{title}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="page-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def render_navigation() -> str:
    active_page = st.session_state.get("active_page", "Home")
    with st.container(key="nav_shell"):
        brand_col, *nav_cols = st.columns([1.25, 1, 1.15, 1.65, 1.45, 0.8], vertical_alignment="center")
        with brand_col:
            st.markdown(
                '<div class="brand-lockup"><span class="brand-mark">M</span>'
                '<span class="brand-name">MEDISYNTH</span></div>',
                unsafe_allow_html=True,
            )
        for column, page in zip(nav_cols, NAV_ITEMS):
            with column:
                if st.button(
                    page,
                    key=f"nav_{page.lower().replace(' ', '_')}",
                    type="primary" if active_page == page else "secondary",
                    use_container_width=True,
                ):
                    st.session_state["active_page"] = page
                    st.rerun()
    return active_page


def render_overview() -> None:
    summary = load_dashboard_csv("dataset_summary.csv").iloc[0]
    performance = load_dashboard_csv("model_performance.csv")
    random_forest = performance.loc[performance["model_name"] == "Random Forest"].iloc[0]

    page_header(
        "Project overview",
        "A completed PySpark machine-learning workflow for heart-disease classification.",
    )
    st.markdown(
        '<div class="scope-note"><strong>Dataset totals</strong> describe the full cleaned cohort. '
        '<strong>Accuracy and ROC-AUC</strong> below are measured on the held-out Phase 4 test set.</div>',
        unsafe_allow_html=True,
    )

    kpis = st.columns(5, gap="medium")
    with kpis[0]:
        metric_card("Total cleaned records", f"{int(summary['total_records']):,}", "Full cleaned dataset")
    with kpis[1]:
        metric_card("No Disease", f"{int(summary['no_disease_count']):,}", "Full cleaned dataset · class 0")
    with kpis[2]:
        metric_card("Heart Disease", f"{int(summary['disease_count']):,}", "Full cleaned dataset · class 1")
    with kpis[3]:
        metric_card("Random Forest accuracy", f"{random_forest['accuracy']:.2%}", "Held-out test set")
    with kpis[4]:
        metric_card("Random Forest ROC-AUC", f"{random_forest['roc_auc']:.2%}", "Held-out test set")

    cohort = load_cleaned_dataset()
    st.markdown("")
    chart_row = st.columns(2, gap="large")
    with chart_row[0]:
        section_heading("Disease distribution", "Observed outcomes in the full cleaned dataset")
        distribution = (
            cohort["cardio"].value_counts().reindex([0, 1], fill_value=0)
            .rename_axis("class").reset_index(name="Records")
        )
        distribution["Outcome"] = distribution["class"].map({0: "No Disease", 1: "Heart Disease"})
        fig = px.pie(
            distribution,
            names="Outcome",
            values="Records",
            hole=0.66,
            color="Outcome",
            color_discrete_map={"No Disease": COLORS["blue_light"], "Heart Disease": COLORS["teal"]},
        )
        fig.update_traces(textinfo="label+value", textposition="outside", textfont_color=COLORS["text"])
        show_chart(fig, key="overview_disease_distribution", height=330)
    with chart_row[1]:
        section_heading("Age distribution", "Age in years · full cleaned dataset")
        fig = px.histogram(cohort, x="age_years", nbins=36, labels={"age_years": "Age (years)", "count": "Records"})
        fig.update_traces(marker_color=COLORS["blue"], marker_line_color=COLORS["card"], marker_line_width=1)
        show_chart(fig, key="overview_age_distribution", height=330)

    chart_row = st.columns(2, gap="large")
    with chart_row[0]:
        section_heading("Cholesterol distribution", "Clinical category values from the cleaned dataset")
        cholesterol = (
            cohort["cholesterol"].value_counts().sort_index()
            .rename_axis("Category").reset_index(name="Records")
        )
        cholesterol["Category"] = cholesterol["Category"].map(
            {1: "1 · Normal", 2: "2 · Above normal", 3: "3 · Well above normal"}
        )
        fig = px.bar(cholesterol, x="Category", y="Records", text="Records", color_discrete_sequence=[COLORS["blue"]])
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        show_chart(fig, key="overview_cholesterol_distribution", height=320)
    with chart_row[1]:
        section_heading("Glucose distribution", "Clinical category values from the cleaned dataset")
        glucose = (
            cohort["gluc"].value_counts().sort_index()
            .rename_axis("Category").reset_index(name="Records")
        )
        glucose["Category"] = glucose["Category"].map(
            {1: "1 · Normal", 2: "2 · Above normal", 3: "3 · Well above normal"}
        )
        fig = px.bar(glucose, x="Category", y="Records", text="Records", color_discrete_sequence=[COLORS["teal"]])
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        show_chart(fig, key="overview_glucose_distribution", height=320)

    section_heading("Blood pressure distribution", "Systolic and diastolic measurements · full cleaned dataset")
    pressure = cohort[["ap_hi", "ap_lo"]].rename(
        columns={"ap_hi": "Systolic", "ap_lo": "Diastolic"}
    ).melt(var_name="Measurement", value_name="Pressure (mmHg)")
    fig = px.histogram(
        pressure,
        x="Pressure (mmHg)",
        color="Measurement",
        nbins=50,
        barmode="overlay",
        opacity=0.78,
        color_discrete_map={"Systolic": COLORS["blue"], "Diastolic": COLORS["teal"]},
    )
    show_chart(fig, key="overview_blood_pressure_distribution", height=330)
    st.markdown(
        '<div class="scope-note">Workflow: clean null/missing values → prepare numerical and categorical features → '
        'train Logistic Regression and Random Forest → evaluate → predict on held-out data.</div>',
        unsafe_allow_html=True,
    )


def render_prediction_page() -> None:
    page_header(
        "Predictions",
        "Estimate a class and probabilities using the saved Random Forest and fitted preprocessing pipeline.",
    )
    st.markdown(
        '<div class="scope-note">Enter measurements to generate a local model estimate. '
        'No patient values are prefilled.</div>',
        unsafe_allow_html=True,
    )
    with st.form("patient_prediction_form"):
        section_heading("Patient features")
        age_col, gender_col, height_col, weight_col = st.columns(4)
        with age_col:
            age = st.number_input("Age (years)", min_value=29, max_value=64, value=None, step=1)
        with gender_col:
            gender = st.selectbox(
                "Gender",
                options=[None, 1, 2],
                format_func=lambda value: "Select…" if value is None else {1: "Female (1)", 2: "Male (2)"}[value],
            )
        with height_col:
            height = st.number_input("Height (cm)", min_value=120, max_value=220, value=None, step=1)
        with weight_col:
            weight = st.number_input("Weight (kg)", min_value=35.0, max_value=200.0, value=None, step=0.5)

        systolic_col, diastolic_col, cholesterol_col, glucose_col = st.columns(4)
        with systolic_col:
            systolic = st.number_input("Systolic blood pressure (mmHg)", min_value=60, max_value=250, value=None, step=1)
        with diastolic_col:
            diastolic = st.number_input("Diastolic blood pressure (mmHg)", min_value=40, max_value=150, value=None, step=1)
        with cholesterol_col:
            cholesterol = st.selectbox(
                "Cholesterol",
                options=[None, 1, 2, 3],
                format_func=lambda value: "Select…" if value is None else {
                    1: "Normal (1)", 2: "Above normal (2)", 3: "Well above normal (3)"
                }[value],
            )
        with glucose_col:
            glucose = st.selectbox(
                "Glucose",
                options=[None, 1, 2, 3],
                format_func=lambda value: "Select…" if value is None else {
                    1: "Normal (1)", 2: "Above normal (2)", 3: "Well above normal (3)"
                }[value],
            )

        smoking_col, alcohol_col, active_col, _ = st.columns(4)
        with smoking_col:
            smoking = st.selectbox(
                "Smoking", options=[None, 0, 1],
                format_func=lambda value: "Select…" if value is None else {0: "No (0)", 1: "Yes (1)"}[value],
            )
        with alcohol_col:
            alcohol = st.selectbox(
                "Alcohol consumption", options=[None, 0, 1],
                format_func=lambda value: "Select…" if value is None else {0: "No (0)", 1: "Yes (1)"}[value],
            )
        with active_col:
            active = st.selectbox(
                "Physical activity", options=[None, 0, 1],
                format_func=lambda value: "Select…" if value is None else {0: "No (0)", 1: "Yes (1)"}[value],
            )
        submitted = st.form_submit_button("Generate model estimate", type="primary")

    if submitted:
        st.session_state.pop("prediction_result", None)
        features = {
            "age_years": age,
            "gender": gender,
            "height": height,
            "weight": weight,
            "ap_hi": systolic,
            "ap_lo": diastolic,
            "cholesterol": cholesterol,
            "gluc": glucose,
            "smoke": smoking,
            "alco": alcohol,
            "active": active,
        }
        try:
            st.session_state["prediction_result"] = load_predictor().predict_patient(features)
        except (TypeError, ValueError) as error:
            st.error(f"Please correct the input values: {error}")
            return
        except Exception as error:
            st.error(f"Prediction failed ({type(error).__name__}): {error}")
            return

    result = st.session_state.get("prediction_result")
    if result is not None:
        predicted_class = int(result["predicted_class"])
        class_label = "Heart Disease" if predicted_class == 1 else "No Disease"
        probability_no_disease = float(result["probability_no_disease"])
        probability_disease = float(result["probability_disease"])
        with st.container(border=True):
            section_heading("Model estimate", "Random Forest · saved model and fitted preprocessing · no retraining")
            result_cols = st.columns(3, gap="medium")
            with result_cols[0]:
                metric_card("Predicted class", f"{predicted_class}: {class_label}", "0 = No Disease | 1 = Heart Disease")
            with result_cols[1]:
                metric_card("No-disease probability", f"{probability_no_disease:.1%}", "Class 0 = No Disease")
            with result_cols[2]:
                metric_card("Disease probability", f"{probability_disease:.1%}", "Class 1 = Heart Disease")
            probability_data = pd.DataFrame(
                {
                    "Class": ["0 = No Disease", "1 = Heart Disease"],
                    "Probability": [probability_no_disease, probability_disease],
                }
            )
            fig = px.bar(
                probability_data,
                x="Class",
                y="Probability",
                color="Class",
                text=probability_data["Probability"].map(lambda value: f"{value:.1%}"),
                color_discrete_map={
                    "0 = No Disease": COLORS["blue_light"],
                    "1 = Heart Disease": COLORS["teal"],
                },
            )
            fig.update_traces(textposition="outside")
            fig.update_yaxes(range=[0, 1], tickformat=".0%")
            fig.update_layout(showlegend=False)
            show_chart(fig, key="prediction_probabilities", height=280)
            st.markdown(
                '<div class="scope-note">This output is an ML estimate, not a medical diagnosis, '
                'and should not replace professional medical assessment.</div>',
                unsafe_allow_html=True,
            )


def render_actual_vs_predicted(predictions: pd.DataFrame) -> None:
    section_heading(
        "Actual vs predicted",
        "Random Forest predictions on the saved Phase 4 test set; counts are aggregated without patient identifiers.",
    )
    rf_predictions = predictions.loc[predictions["model_name"] == "Random Forest"]
    actual_counts = rf_predictions["actual_class"].value_counts().reindex([0, 1], fill_value=0)
    predicted_counts = rf_predictions["predicted_class"].value_counts().reindex([0, 1], fill_value=0)
    actual_correct = int((rf_predictions["actual_class"] == rf_predictions["predicted_class"]).sum())
    incorrect = int(len(rf_predictions) - actual_correct)
    cards = st.columns(4, gap="medium")
    with cards[0]:
        metric_card("Actual · No Disease", f"{int(actual_counts[0]):,}", "Saved test predictions")
    with cards[1]:
        metric_card("Actual · Heart Disease", f"{int(actual_counts[1]):,}", "Saved test predictions")
    with cards[2]:
        metric_card("Correct predictions", f"{actual_correct:,}", "Random Forest · held-out test set")
    with cards[3]:
        metric_card("Incorrect predictions", f"{incorrect:,}", "Random Forest · held-out test set")

    comparison = pd.DataFrame(
        {
            "Class": ["0 = No Disease", "1 = Heart Disease"],
            "Actual": [int(actual_counts[0]), int(actual_counts[1])],
            "Predicted": [int(predicted_counts[0]), int(predicted_counts[1])],
        }
    ).melt(id_vars="Class", var_name="Series", value_name="Records")
    fig = px.bar(
        comparison,
        x="Class",
        y="Records",
        color="Series",
        barmode="group",
        text="Records",
        color_discrete_map={"Actual": COLORS["blue"], "Predicted": COLORS["teal"]},
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    show_chart(fig, key="actual_predicted_comparison", height=350)


def render_model_evaluation() -> None:
    page_header(
        "Model evaluation",
        "Compare Logistic Regression and Random Forest using the saved Phase 4 held-out test results.",
    )
    performance = load_dashboard_csv("model_performance.csv")
    section_heading("Evaluation metrics", "Accuracy, precision, recall, F1-score and ROC-AUC")
    display_metrics = performance.rename(
        columns={
            "model_name": "Model",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1-score",
            "roc_auc": "ROC-AUC",
        }
    )
    st.dataframe(
        display_metrics.style.format(
            {name: "{:.4f}" for name in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]}
        ),
        hide_index=True,
        use_container_width=True,
    )
    metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_data = performance.melt(
        id_vars="model_name",
        value_vars=metric_names,
        var_name="Metric",
        value_name="Score",
    )
    metric_data["Metric"] = metric_data["Metric"].map(
        {"f1_score": "F1-score", "roc_auc": "ROC-AUC"}
    ).fillna(metric_data["Metric"].str.title())
    metric_data["Metric"] = pd.Categorical(
        metric_data["Metric"],
        categories=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
        ordered=True,
    )
    section_heading("Model comparison", "Grouped bars show the saved Phase 4 metrics")
    fig = px.bar(
        metric_data,
        x="Metric",
        y="Score",
        color="model_name",
        barmode="group",
        color_discrete_map=MODEL_COLORS,
        labels={"model_name": "Model", "Score": "Test-set score"},
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    show_chart(fig, key="model_metric_comparison", height=360)

    predictions = load_dashboard_csv("predictions.csv")
    render_actual_vs_predicted(predictions)

    roc = load_dashboard_csv("roc_curve.csv")
    section_heading("ROC curve comparison", "Saved Phase 4 ROC points · diagonal represents random classification")
    fig = go.Figure()
    for model_name in ["Logistic Regression", "Random Forest"]:
        model_roc = roc.loc[roc["model_name"] == model_name].sort_values("false_positive_rate")
        auc = float(performance.loc[performance["model_name"] == model_name, "roc_auc"].iloc[0])
        fig.add_trace(
            go.Scatter(
                x=model_roc["false_positive_rate"],
                y=model_roc["true_positive_rate"],
                mode="lines",
                name=f"{model_name} (AUC = {auc:.4f})",
                line={"color": MODEL_COLORS[model_name], "width": 3},
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random classifier",
            line={"color": "#94A3B8", "width": 1.5, "dash": "dash"},
        )
    )
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1])
    show_chart(fig, key="roc_curve_comparison", height=390)


def confusion_figure(matrix: pd.DataFrame, model_name: str) -> go.Figure:
    counts = (
        matrix.loc[matrix["model_name"] == model_name]
        .pivot(index="actual_class", columns="predicted_class", values="count")
        .reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    )
    cells = [
        [f"TN<br>{int(counts.loc[0, 0]):,}", f"FP<br>{int(counts.loc[0, 1]):,}"],
        [f"FN<br>{int(counts.loc[1, 0]):,}", f"TP<br>{int(counts.loc[1, 1]):,}"],
    ]
    fig = go.Figure(
        go.Heatmap(
            z=counts.values,
            x=["0 = No Disease", "1 = Heart Disease"],
            y=["0 = No Disease", "1 = Heart Disease"],
            text=cells,
            texttemplate="%{text}",
            textfont={"size": 16, "color": COLORS["text"]},
            colorscale=[[0, "#DBEAFE"], [0.55, "#93C5FD"], [1, "#3B82F6"]],
            showscale=False,
            xgap=5,
            ygap=5,
        )
    )
    fig.update_layout(title={"text": model_name, "x": 0.02})
    fig.update_xaxes(title="Predicted class")
    fig.update_yaxes(title="Actual class", autorange="reversed")
    return fig


def render_confusion_matrix() -> None:
    page_header(
        "Confusion matrix",
        "Saved Phase 4 test counts. Rows are actual classes; columns are predicted classes.",
    )
    matrix = load_dashboard_csv("confusion_matrix.csv")
    chart_cols = st.columns(2, gap="large")
    for column, model_name, key in zip(
        chart_cols,
        ["Logistic Regression", "Random Forest"],
        ["confusion_logistic", "confusion_random_forest"],
    ):
        with column:
            section_heading(model_name, "TN · true negative | FP · false positive | FN · false negative | TP · true positive")
            show_chart(confusion_figure(matrix, model_name), key=key, height=380)


def render_reports() -> None:
    page_header(
        "Reports",
        "Download the existing Phase 4 and Phase 5 project outputs. No reports are regenerated here.",
    )
    section_heading("Project CSV files", "Downloads contain the saved project records and evaluation results.")
    for start in range(0, len(REPORT_FILES), 3):
        columns = st.columns(3, gap="medium")
        for column, filename in zip(columns, REPORT_FILES[start : start + 3]):
            with column:
                path = DASHBOARD_DATA / filename
                if not path.is_file():
                    st.error(f"Required report is missing: {path}")
                    continue
                column_name = filename.removesuffix(".csv").replace("_", " ").title()
                st.download_button(
                    f"Download {column_name} CSV",
                    data=load_report_bytes(filename),
                    file_name=filename,
                    mime="text/csv",
                    key=f"download_{filename}",
                    use_container_width=True,
                )
                st.caption(f"{path.stat().st_size:,} bytes · existing saved report")


def main() -> None:
    load_css()
    active_page = render_navigation()
    if active_page == "Home":
        render_overview()
    elif active_page == "Predictions":
        render_prediction_page()
    elif active_page == "Model Evaluation":
        render_model_evaluation()
    elif active_page == "Confusion Matrix":
        render_confusion_matrix()
    elif active_page == "Reports":
        render_reports()


if __name__ == "__main__":
    main()
