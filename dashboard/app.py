# =========================================================
# APP.PY
# DASHBOARD STANCE DETECTION FOOD ESTATE
# =========================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import plotly.graph_objects as go

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from preprocessing import preprocess_text

from results_data import (
    RESULTS,
    CLASS_RESULTS,
    LABELS,
    MODEL_CONFIG
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Stance Detection",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# PATH
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


SVM_BASELINE_PATH = os.path.join(
    MODEL_DIR,
    "svm_baseline_calibrated.pkl"
)

SVM_HPO_PATH = os.path.join(
    MODEL_DIR,
    "svm_hpo_calibrated.pkl"
)

INDOBERT_BASE_PATH = os.path.join(
    MODEL_DIR,
    "indobert_baseline"
)

INDOBERT_HPO_PATH = os.path.join(
    MODEL_DIR,
    "indobert_hpo"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #f5f7fb;
        color: #1f2937 !important;
        text-align: center;
        border: 1px solid #e5e7eb;
    }

    /* =====================================================
        KESIMPULAN
       ===================================================== */

    .conclusion-box {
        padding: 25px;
        border-radius: 12px;
        background-color: #f5f7fb !important;
        color: #1f2937 !important;
        border-left: 5px solid #4f6fdc;
        margin-top: 15px;
        margin-bottom: 15px;
        line-height: 1.7;
    }

    .conclusion-box b {
        color: #1f2937 !important;
    }

    .conclusion-box strong {
        color: #1f2937 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📌 Navigasi")

page = st.sidebar.radio(
    "Pilih Halaman",
    [
        "📊 Informasi Model",
        "🔍 Prediksi Stance"
    ]
)


st.sidebar.markdown("---")

st.sidebar.caption(
    "Dashboard Stance Detection Opini Publik "
    "terhadap Kebijakan Food Estate"
)


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_svm_baseline():

    return joblib.load(
        SVM_BASELINE_PATH
    )


@st.cache_resource
def load_svm_hpo():

    return joblib.load(
        SVM_HPO_PATH
    )


@st.cache_resource
def load_indobert(path):

    tokenizer = AutoTokenizer.from_pretrained(
        path,
        local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        path,
        local_files_only=True
    )

    model.eval()

    return tokenizer, model


# =========================================================
# FORMAT PERCENTAGE
# =========================================================

def percentage(value):

    return f"{value * 100:.2f}%"


# =========================================================
# BAR CHART
# =========================================================

def create_metric_chart():

    models = list(RESULTS.keys())

    metrics = [
        ("Accuracy", "accuracy"),
        ("Precision (weighted)", "precision_weighted"),
        ("Recall (weighted)", "recall_weighted"),
        ("F1 (weighted)", "f1_weighted")
    ]

    fig = go.Figure()

    colors = [
        "#2F5BEA",
        "#7C9AEF",
        "#F2A033",
        "#F8D052"
    ]

    for i, model_name in enumerate(models):

        values = [
            RESULTS[model_name][key]
            for _, key in metrics
        ]

        fig.add_trace(
            go.Bar(
                name=model_name,
                x=[name for name, _ in metrics],
                y=values,
                text=[
                    f"{v * 100:.2f}%"
                    for v in values
                ],
                textposition="outside",
                marker_color=colors[i]
            )
        )

    fig.update_layout(
        title="Perbandingan Kinerja Empat Model",
        yaxis=dict(
            title="Nilai",
            range=[0, 0.85],
            tickformat=".0%"
        ),
        xaxis_title="Metrik",
        barmode="group",
        height=500,
        legend_title="Model",
        template="plotly_white"
    )

    return fig


# =========================================================
# F1 PER KELAS
# =========================================================

def create_class_f1_chart(model_name):

    base_name = model_name

    if "IndoBERT" in model_name:

        base_model = "IndoBERT (Base)"
        hpo_model = "IndoBERT (HPO)"

    else:

        base_model = "SVM (Base)"
        hpo_model = "SVM (HPO)"

    base_values = [
        CLASS_RESULTS[base_model][label]["f1"]
        for label in LABELS
    ]

    hpo_values = [
        CLASS_RESULTS[hpo_model][label]["f1"]
        for label in LABELS
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Base",
            x=LABELS,
            y=base_values,
            text=[
                f"{v:.3f}"
                for v in base_values
            ],
            textposition="inside",
            marker_color="#5B7CDB"
        )
    )

    fig.add_trace(
        go.Bar(
            name="HPO",
            x=LABELS,
            y=hpo_values,
            text=[
                f"{v:.3f}"
                for v in hpo_values
            ],
            textposition="inside",
            marker_color="#9B83DC"
        )
    )

    fig.update_layout(
        title=f"Perbandingan F1-Score per Kelas — {base_model.replace(' (Base)', '')}",
        yaxis=dict(
            title="F1-Score",
            range=[0, 1]
        ),
        xaxis_title="Kelas",
        barmode="group",
        height=450,
        template="plotly_white",
        legend_title="Model"
    )

    return fig


# =========================================================
# F1 PER KELAS — DUA GRAFIK BERDAMPINGAN
# =========================================================

def create_side_by_side_class_f1():

    fig = go.Figure()

    for model_type, base_name, hpo_name in [
        (
            "IndoBERT",
            "IndoBERT (Base)",
            "IndoBERT (HPO)"
        ),
        (
            "SVM",
            "SVM (Base)",
            "SVM (HPO)"
        )
    ]:

        for label, color, model_name in [
            (
                "Base",
                "#5B7CDB",
                base_name
            ),
            (
                "HPO",
                "#9B83DC",
                hpo_name
            )
        ]:

            fig.add_trace(
                go.Bar(
                    name=f"{model_type} - {label}",
                    x=[
                        f"{model_type}<br>{class_name}"
                        for class_name in LABELS
                    ],
                    y=[
                        CLASS_RESULTS[
                            model_name
                        ][class_name]["f1"]
                        for class_name in LABELS
                    ],
                    text=[
                        f"{CLASS_RESULTS[model_name][class_name]['f1']:.3f}"
                        for class_name in LABELS
                    ],
                    textposition="inside"
                )
            )

    fig.update_layout(
        title="Perbandingan F1-Score per Kelas",
        yaxis=dict(
            title="F1-Score",
            range=[0, 1]
        ),
        barmode="group",
        height=500,
        template="plotly_white"
    )

    return fig


# =========================================================
# CLASS RESULT TABLE
# =========================================================

def create_class_table(model_name):

    rows = []

    for label in LABELS:

        data = CLASS_RESULTS[
            model_name
        ][label]

        rows.append(
            {
                "Kelas": label,
                "Precision": data["precision"],
                "Recall": data["recall"],
                "F1-Score": data["f1"]
            }
        )

    df = pd.DataFrame(rows)

    for column in [
        "Precision",
        "Recall",
        "F1-Score"
    ]:

        df[column] = df[column].map(
            lambda x: f"{x:.4f}"
        )

    return df


# =========================================================
# CONFUSION MATRIX
# =========================================================

def load_confusion_matrix(filename):

    path = os.path.join(
        RESULTS_DIR,
        filename
    )

    if not os.path.exists(path):

        return None

    try:

        df = pd.read_csv(
            path,
            index_col=0
        )

        return df

    except Exception:

        return None


def create_confusion_matrix(
    df,
    title
):

    if df is None:

        return None

    labels = list(df.index)

    values = df.values

    fig = go.Figure(
        data=go.Heatmap(
            z=values,
            x=labels,
            y=labels,
            colorscale="viridis",
            text=values,
            texttemplate="%{text}",
            textfont={
                "size": 16
            },
            hovertemplate=(
                "Aktual: %{y}<br>"
                "Prediksi: %{x}<br>"
                "Jumlah: %{z}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Prediksi",
        yaxis_title="Aktual",
        height=450,
        template="plotly_white"
    )

    return fig


# =========================================================
# PAGE 1
# INFORMASI MODEL
# =========================================================

if page == "📊 Informasi Model":

    st.markdown(
        '<div class="main-title">Stance Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Dashboard evaluasi model SVM dan IndoBERT '
        'untuk klasifikasi stance opini publik terhadap '
        'kebijakan Food Estate.'
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # DATASET
    # =====================================================

    st.header("📁 Informasi Dataset")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Jumlah Data",
            "1.010"
        )

    with col2:

        st.metric(
            "Pro",
            "376"
        )

    with col3:

        st.metric(
            "Kontra",
            "382"
        )

    with col4:

        st.metric(
            "Netral",
            "253"
        )


    st.info(
        """
        Dataset terdiri dari komentar publik yang dikumpulkan
        dari YouTube dan X mengenai kebijakan Food Estate.
        Data diklasifikasikan ke dalam tiga kelas stance:
        Pro, Kontra, dan Netral.
        """
    )


    # =====================================================
    # HASIL KESELURUHAN
    # =====================================================

    st.header("📊 Perbandingan Hasil Model")

    st.plotly_chart(
        create_metric_chart(),
        use_container_width=True
    )


    # =====================================================
    # TABEL HASIL
    # =====================================================

    st.subheader("Tabel Hasil Evaluasi")

    result_rows = []

    for model_name, data in RESULTS.items():

        result_rows.append(
            {
                "Model": model_name,
                "Accuracy": percentage(
                    data["accuracy"]
                ),
                "Precision (weighted)": percentage(
                    data["precision_weighted"]
                ),
                "Recall (weighted)": percentage(
                    data["recall_weighted"]
                ),
                "F1 (weighted)": percentage(
                    data["f1_weighted"]
                )
            }
        )

    result_df = pd.DataFrame(
        result_rows
    )

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # MODEL TERBAIK
    # =====================================================

    best_model = max(
        RESULTS,
        key=lambda x:
        RESULTS[x]["f1_weighted"]
    )

    st.success(
        f"Model dengan F1-Score weighted tertinggi "
        f"adalah **{best_model}** dengan nilai "
        f"**{RESULTS[best_model]['f1_weighted']:.2%}**."
    )

    # =====================================================
    # F1 PER KELAS — TABEL
    # =====================================================
    st.header(
        "📈 Perbandingan F1-Score per Kelas"
    )
    st.subheader(
        "Detail F1-Score per Kelas"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### IndoBERT"
        )

        indobert_rows = []

        for label in LABELS:

            base = CLASS_RESULTS[
                "IndoBERT (Base)"
            ][label]["f1"]

            hpo = CLASS_RESULTS[
                "IndoBERT (HPO)"
            ][label]["f1"]

            indobert_rows.append(
                {
                    "Kelas": label,
                    "Base": f"{base:.4f}",
                    "HPO": f"{hpo:.4f}",
                    "Δ F1": f"{hpo - base:+.4f}"
                }
            )

        st.dataframe(
            pd.DataFrame(indobert_rows),
            use_container_width=True,
            hide_index=True
        )


    with col2:

        st.markdown(
            "### SVM"
        )

        svm_rows = []

        for label in LABELS:

            base = CLASS_RESULTS[
                "SVM (Base)"
            ][label]["f1"]

            hpo = CLASS_RESULTS[
                "SVM (HPO)"
            ][label]["f1"]

            svm_rows.append(
                {
                    "Kelas": label,
                    "Base": f"{base:.4f}",
                    "HPO": f"{hpo:.4f}",
                    "Δ F1": f"{hpo - base:+.4f}"
                }
            )

        st.dataframe(
            pd.DataFrame(svm_rows),
            use_container_width=True,
            hide_index=True
        )


    # =====================================================
    # F1 PER KELAS
    # =====================================================

    st.plotly_chart(
        create_side_by_side_class_f1(),
        use_container_width=True
    )


    # =====================================================
    # ANALISIS PERBANDINGAN
    # =====================================================

    st.header(
        "🔎 Analisis Perbandingan"
    )


    indobert_base_f1 = RESULTS[
        "IndoBERT (Base)"
    ]["f1_weighted"]

    svm_base_f1 = RESULTS[
        "SVM (Base)"
    ]["f1_weighted"]

    indobert_hpo_f1 = RESULTS[
        "IndoBERT (HPO)"
    ]["f1_weighted"]

    svm_hpo_f1 = RESULTS[
        "SVM (HPO)"
    ]["f1_weighted"]


    gap_base = (
        indobert_base_f1 -
        svm_base_f1
    )

    gap_hpo = (
        indobert_hpo_f1 -
        svm_hpo_f1
    )


    indobert_change = (
        indobert_hpo_f1 -
        indobert_base_f1
    )

    svm_change = (
        svm_hpo_f1 -
        svm_base_f1
    )


    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "Gap IndoBERT vs SVM (F1 Weighted)"
        )

        st.metric(
            "Base — Selisih F1",
            f"{gap_base:+.2%}"
        )

        st.metric(
            "HPO — Selisih F1",
            f"{gap_hpo:+.2%}"
        )


    with col2:

        st.subheader(
            "Selisih F1-Score (w) (HPO - Base)"
        )

        st.metric(
            "IndoBERT",
            f"{indobert_change:+.2%}"
        )

        st.metric(
            "SVM",
            f"{svm_change:+.2%}"
        )

    # =========================================================
    # KESIMPULAN
    # =========================================================
    st.header("📝 Kesimpulan")

    # CSS untuk tampilan kartu
    st.markdown(
        """
        <style>
        .st-key-conclusion_card {
            background-color: #F3F6FC;
            border: 1px solid #DCE5F5;
            border-left: 6px solid #4F6FE8;
            border-radius: 14px;
            padding: 24px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Perbandingan model
    if gap_hpo > 0:
        model_comparison = (
            "IndoBERT menunjukkan performa yang lebih unggul "
            "dibandingkan SVM pada skema optimasi hyperparameter, dengan selisih "
            f"F1-Score weighted sebesar **{gap_hpo:.2%}**."
        )
    else:
        model_comparison = (
            "SVM menunjukkan performa yang lebih unggul "
            "dibandingkan IndoBERT pada skema HPO, dengan selisih "
            f"F1-Score weighted sebesar **{abs(gap_hpo):.2%}**."
        )

    # Pengaruh HPO terhadap SVM
    if svm_change > 0:
        svm_statement = (
            "Optimasi hyperparameter meningkatkan performa "
            f"SVM sebesar **{svm_change:.2%}** pada F1-Score weighted."
        )
    elif svm_change < 0:
        svm_statement = (
            "Optimasi hyperparameter menurunkan performa SVM "
            f"sebesar **{abs(svm_change):.2%}** pada F1-Score weighted."
        )
    else:
        svm_statement = (
            "Optimasi hyperparameter tidak mengubah "
            "F1-Score weighted SVM."
        )

    # Pengaruh HPO terhadap IndoBERT
    if indobert_change > 0:
        indobert_statement = (
            "Pada IndoBERT, optimasi hyperparameter meningkatkan F1-Score weighted "
            f"sebesar **{indobert_change:.2%}**."
        )
    elif indobert_change < 0:
        indobert_statement = (
            "Pada IndoBERT, HPO menurunkan F1-Score weighted "
            f"sebesar **{abs(indobert_change):.2%}**."
        )
    else:
        indobert_statement = (
            "Pada IndoBERT, HPO tidak mengubah "
            "F1-Score weighted."
        )

    # Bagian 1 — Perbandingan Model
    with st.container(border=True):
        st.subheader("🏆 Perbandingan Model")
        st.markdown(model_comparison)

        st.divider()

        # Bagian 2 — Pengaruh HPO
        st.subheader("⚙️ Pengaruh Hyperparameter Optimization")
        st.markdown(indobert_statement)
        st.markdown(svm_statement)

        st.divider()

        # Bagian 3 — Performa per Kelas
        st.subheader("📊 Performa Per Kelas")
        st.markdown(
            "F1-Score kelas **Netral** menjadi salah satu aspek "
            "penting dalam evaluasi karena kelas ini memiliki "
            "performa yang lebih rendah dibandingkan kelas Pro "
            "dan Kontra pada keempat model."
        )


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    st.header(
        "🔲 Confusion Matrix"
    )

    st.caption(
        "Confusion matrix ditampilkan berdasarkan hasil "
        "Out-of-Fold prediction dari Stratified 5-Fold."
    )


    cm_files = {

        "IndoBERT (Base)": (
            "indobert_baseline_confusion_matrix.csv"
        ),

        "IndoBERT (HPO)": (
            "indobert_hpo_confusion_matrix.csv"
        ),

        "SVM (Base)": (
            "svm_baseline_confusion_matrix.csv"
        ),

        "SVM (HPO)": (
            "svm_hpo_confusion_matrix.csv"
        )
    }


    cm_col1, cm_col2 = st.columns(2)


    for i, (model_name, filename) in enumerate(
        cm_files.items()
    ):

        cm = load_confusion_matrix(
            filename
        )

        fig = create_confusion_matrix(
            cm,
            f"Confusion Matrix — {model_name}"
        )


        if i % 2 == 0:

            current_col = cm_col1

        else:

            current_col = cm_col2


        with current_col:

            if fig is not None:

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.warning(
                    f"File confusion matrix untuk "
                    f"{model_name} belum ditemukan."
                )


    # =====================================================
    # KONFIGURASI MODEL
    # =====================================================

    st.header(
        "⚙️ Konfigurasi Model"
    )


    selected_config_model = st.selectbox(
        "Pilih model",
        list(MODEL_CONFIG.keys())
    )


    config_data = MODEL_CONFIG[
        selected_config_model
    ]


    config_rows = []

    for key, value in config_data.items():

        config_rows.append(
            {
                "Parameter": key,
                "Nilai": value
            }
        )


    st.dataframe(
        pd.DataFrame(config_rows),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# PAGE 2
# PREDIKSI
# =========================================================

elif page == "🔍 Prediksi Stance":

    st.title(
        "🔍 Prediksi Stance"
    )

    st.write(
        "Masukkan komentar publik untuk memperoleh "
        "prediksi stance dari empat model."
    )


    # =====================================================
    # INPUT
    # =====================================================

    text_input = st.text_area(
        "Masukkan komentar:",
        height=180,
        placeholder=(
            "Contoh: Kebijakan Food Estate perlu "
            "dievaluasi agar tidak merugikan masyarakat."
        )
    )


    predict_button = st.button(
        "🚀 Prediksi",
        type="primary"
    )


    # =====================================================
    # PREDICTION
    # =====================================================

    if predict_button:

        if not text_input.strip():

            st.warning(
                "Silakan masukkan komentar terlebih dahulu."
            )

            st.stop()


        raw_text = text_input.strip()


        # =================================================
        # PREPROCESSING UNTUK INFORMASI
        # =================================================

        processed_text = preprocess_text(
            raw_text
        )


        with st.expander(
            "Lihat hasil preprocessing"
        ):

            st.write(
                processed_text
            )


        # =================================================
        # LOAD MODELS
        # =================================================

        with st.spinner(
            "Memuat model..."
        ):

            svm_base = load_svm_baseline()

            svm_hpo = load_svm_hpo()

            tokenizer_base, indobert_base = (
                load_indobert(
                    INDOBERT_BASE_PATH
                )
            )

            tokenizer_hpo, indobert_hpo = (
                load_indobert(
                    INDOBERT_HPO_PATH
                )
            )


        # =================================================
        # LABEL MAP
        # =================================================

        svm_label_map = {
            0: "Pro",
            2: "Kontra",
            3: "Netral"
        }


        indobert_label_map = {
            0: "Pro",
            1: "Kontra",
            2: "Netral"
        }


        # =================================================
        # SVM PREDICTION
        # =================================================

        def predict_svm(
            model,
            text
        ):

            prediction = model.predict(
                [text]
            )[0]

            probabilities = model.predict_proba(
                [text]
            )[0]

            classes = model.classes_

            probability_dict = {}

            for class_id, probability in zip(
                classes,
                probabilities
            ):

                label = svm_label_map.get(
                    class_id,
                    str(class_id)
                )

                probability_dict[
                    label
                ] = probability


            predicted_label = svm_label_map.get(
                prediction,
                str(prediction)
            )


            confidence = probability_dict[
                predicted_label
            ]


            return (
                predicted_label,
                confidence,
                probability_dict
            )


        # =================================================
        # INDOBERT PREDICTION
        # =================================================

        def predict_indobert(
            tokenizer,
            model,
            text
        ):

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )


            with torch.no_grad():

                outputs = model(
                    **inputs
                )


            logits = outputs.logits[0]

            probabilities = torch.softmax(
                logits,
                dim=0
            ).cpu().numpy()


            prediction = int(
                np.argmax(
                    probabilities
                )
            )


            probability_dict = {}

            for class_id, probability in enumerate(
                probabilities
            ):

                label = indobert_label_map[
                    class_id
                ]

                probability_dict[
                    label
                ] = float(
                    probability
                )


            predicted_label = indobert_label_map[
                prediction
            ]


            confidence = probability_dict[
                predicted_label
            ]


            return (
                predicted_label,
                confidence,
                probability_dict
            )


        # =================================================
        # RUN PREDICTION
        # =================================================

        svm_base_result = predict_svm(
            svm_base,
            raw_text
        )

        svm_hpo_result = predict_svm(
            svm_hpo,
            raw_text
        )


        indobert_base_result = predict_indobert(
            tokenizer_base,
            indobert_base,
            raw_text
        )


        indobert_hpo_result = predict_indobert(
            tokenizer_hpo,
            indobert_hpo,
            raw_text
        )


        predictions = {

            "IndoBERT (Base)":
                indobert_base_result,

            "IndoBERT (HPO)":
                indobert_hpo_result,

            "SVM (Base)":
                svm_base_result,

            "SVM (HPO)":
                svm_hpo_result
        }


        # =================================================
        # HASIL PREDIKSI
        # =================================================

        st.header(
            "Hasil Prediksi"
        )


        cols = st.columns(4)


        for col, (
            model_name,
            result
        ) in zip(
            cols,
            predictions.items()
        ):

            label = result[0]

            confidence = result[1]


            with col:

                st.markdown(
                    f"### {model_name}"
                )

                st.metric(
                    "Stance",
                    label
                )

                st.metric(
                    "Confidence",
                    f"{confidence:.2%}"
                )


        # =================================================
        # PROBABILITAS PER KELAS
        # =================================================

        st.header(
            "📊 Confidence Score per Kelas"
        )


        probability_rows = []


        for model_name, result in predictions.items():

            probability_dict = result[2]

            probability_rows.append(
                {
                    "Model": model_name,
                    "Pro": probability_dict.get(
                        "Pro",
                        0
                    ),
                    "Kontra": probability_dict.get(
                        "Kontra",
                        0
                    ),
                    "Netral": probability_dict.get(
                        "Netral",
                        0
                    )
                }
            )


        probability_df = pd.DataFrame(
            probability_rows
        )


        display_df = probability_df.copy()


        for label in LABELS:

            display_df[label] = display_df[
                label
            ].map(
                lambda x: f"{x:.2%}"
            )


        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # BAR CHART CONFIDENCE
        # =================================================

        fig = go.Figure()


        for label in LABELS:

            fig.add_trace(
                go.Bar(
                    name=label,
                    x=probability_df[
                        "Model"
                    ],
                    y=probability_df[
                        label
                    ],
                    text=[
                        f"{v:.2%}"
                        for v in probability_df[
                            label
                        ]
                    ],
                    textposition="outside"
                )
            )


        fig.update_layout(
            title="Perbandingan Confidence Score",
            yaxis=dict(
                title="Probability",
                range=[0, 1],
                tickformat=".0%"
            ),
            xaxis_title="Model",
            barmode="group",
            height=500,
            template="plotly_white"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # =================================================
        # INTERPRETASI
        # =================================================

        st.header(
            "📝 Interpretasi"
        )


        predicted_labels = [
            result[0]
            for result in predictions.values()
        ]


        majority = max(
            set(predicted_labels),
            key=predicted_labels.count
        )


        count_majority = predicted_labels.count(
            majority
        )


        st.info(
            f"Dari empat model, **{count_majority} "
            f"model** memprediksi stance **{majority}**."
        )


        st.caption(
            "Confidence SVM diperoleh dari probabilitas "
            "hasil kalibrasi CalibratedClassifierCV, sedangkan "
            "confidence IndoBERT diperoleh dengan menerapkan "
            "Softmax pada logits model."
        )