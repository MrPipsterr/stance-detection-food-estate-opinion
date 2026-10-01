import streamlit as st
import joblib
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from preprocessing import preprocess_text


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Stance Detection",
    page_icon="",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("Stance Detection")

st.write(
    "Dashboard untuk mendeteksi stance opini publik "
    "terhadap kebijakan Food Estate."
)


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_svm_baseline():
    return joblib.load("models/svm_baseline.pkl")


@st.cache_resource
def load_svm_hpo():
    return joblib.load("models/svm_hpo.pkl")


@st.cache_resource
def load_indobert_baseline():
    model_path = "models/indobert_baseline"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    model.eval()

    return tokenizer, model


@st.cache_resource
def load_indobert_hpo():
    model_path = "models/indobert_hpo"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    model.eval()

    return tokenizer, model


# =========================================================
# LOAD ALL MODELS
# =========================================================

svm_baseline = load_svm_baseline()
svm_hpo = load_svm_hpo()

tokenizer_baseline, indobert_baseline = load_indobert_baseline()
tokenizer_hpo, indobert_hpo = load_indobert_hpo()


# =========================================================
# LABEL MAPPING
# =========================================================

# SVM menggunakan mapping sesuai notebook SVM
svm_label_map = {
    0: "Pro",
    2: "Kontra",
    3: "Netral"
}


# IndoBERT menggunakan mapping sesuai notebook IndoBERT
indobert_label_map = {
    0: "Pro",
    1: "Kontra",
    2: "Netral"
}


# =========================================================
# SVM PREDICTION
# =========================================================

def predict_svm(model, text, label_map):
    """
    Prediksi SVM menggunakan raw text.

    Confidence dihitung dari decision_function,
    kemudian dinormalisasi menggunakan softmax.
    """

    # Prediksi kelas
    prediction_raw = model.predict([text])[0]

    # Decision score
    scores = model.decision_function([text])[0]

    # Softmax terhadap decision score
    exp_scores = np.exp(
        scores - np.max(scores)
    )

    percentages = (
        exp_scores / exp_scores.sum()
    ) * 100

    # Kelas yang digunakan model
    classes = model.classes_

    # Posisi kelas hasil prediksi
    predicted_index = list(classes).index(
        prediction_raw
    )

    # Confidence kelas prediksi
    confidence = percentages[
        predicted_index
    ]

    # Confidence masing-masing kelas
    class_confidence = {}

    for cls, percentage in zip(
        classes,
        percentages
    ):
        label = label_map.get(
            cls,
            str(cls)
        )

        class_confidence[label] = percentage

    # Label prediksi
    prediction = label_map.get(
        prediction_raw,
        str(prediction_raw)
    )

    return (
        prediction,
        confidence,
        class_confidence,
        scores
    )


# =========================================================
# INDOBERT PREDICTION
# =========================================================

def predict_indobert(
    model,
    tokenizer,
    text,
    label_map,
    max_length=128
):
    """
    Prediksi IndoBERT dengan alur tokenisasi
    yang disesuaikan dengan notebook.

    Tokenisasi:
    - padding="max_length"
    - truncation=True
    - max_length=128

    Output:
    logits -> softmax -> argmax
    """

    # -----------------------------------------------------
    # TOKENIZATION
    # -----------------------------------------------------

    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    with torch.no_grad():

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    # -----------------------------------------------------
    # LOGITS
    # -----------------------------------------------------

    logits = outputs.logits[0]

    # -----------------------------------------------------
    # SOFTMAX
    # -----------------------------------------------------

    probs = torch.softmax(
        logits,
        dim=0
    ).cpu().numpy()

    # -----------------------------------------------------
    # PREDICTED CLASS
    # -----------------------------------------------------

    pred = int(
        np.argmax(probs)
    )

    prediction = label_map.get(
        pred,
        str(pred)
    )

    # -----------------------------------------------------
    # CONFIDENCE PER CLASS
    # -----------------------------------------------------

    class_confidence = {}

    for cls, probability in enumerate(probs):

        label = label_map.get(
            cls,
            str(cls)
        )

        class_confidence[label] = (
            probability * 100
        )

    # Confidence kelas prediksi
    confidence = (
        probs[pred] * 100
    )

    return (
        prediction,
        confidence,
        class_confidence,
        logits.cpu().numpy()
    )


# =========================================================
# INPUT KOMENTAR
# =========================================================

st.subheader("Input Komentar")

text = st.text_area(
    "Masukkan komentar:",
    placeholder=(
        "Contoh: Kebijakan Food Estate sangat "
        "membantu ketahanan pangan Indonesia."
    ),
    height=150
)


# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button(
    "Prediksi",
    type="primary",
    use_container_width=True
):

    # -----------------------------------------------------
    # VALIDASI INPUT
    # -----------------------------------------------------

    if not text.strip():

        st.warning(
            "Silakan masukkan komentar terlebih dahulu."
        )

        st.stop()


    # -----------------------------------------------------
    # RAW TEXT
    # -----------------------------------------------------

    raw_text = text


    # -----------------------------------------------------
    # PREPROCESSING
    # -----------------------------------------------------
    #
    # Preprocessing tetap ditampilkan di dashboard
    # sebagai informasi kepada pengguna.
    #
    # Namun sesuai notebook prediksi model,
    # model menerima RAW TEXT.
    #

    processed_text = preprocess_text(
        raw_text
    )


    # =====================================================
    # DISPLAY INPUT
    # =====================================================

    st.divider()

    st.subheader("Teks Input")

    st.write(raw_text)


    st.subheader("Hasil Preprocessing")

    st.code(
        processed_text,
        language=None
    )


    # =====================================================
    # MODEL INPUT
    # =====================================================

    # Sesuai notebook:
    # SVM menggunakan raw text
    svm_input = raw_text

    # Sesuai notebook:
    # IndoBERT menggunakan raw text
    indobert_input = raw_text


    # =====================================================
    # SVM BASELINE
    # =====================================================

    (
        svm_baseline_prediction,
        svm_baseline_confidence,
        svm_baseline_class_confidence,
        svm_baseline_scores
    ) = predict_svm(
        svm_baseline,
        svm_input,
        svm_label_map
    )


    # =====================================================
    # SVM HPO
    # =====================================================

    (
        svm_hpo_prediction,
        svm_hpo_confidence,
        svm_hpo_class_confidence,
        svm_hpo_scores
    ) = predict_svm(
        svm_hpo,
        svm_input,
        svm_label_map
    )


    # =====================================================
    # INDOBERT BASELINE
    # =====================================================

    (
        indobert_baseline_prediction,
        indobert_baseline_confidence,
        indobert_baseline_class_confidence,
        indobert_baseline_scores
    ) = predict_indobert(
        indobert_baseline,
        tokenizer_baseline,
        indobert_input,
        indobert_label_map,
        max_length=128
    )


    # =====================================================
    # INDOBERT HPO
    # =====================================================

    (
        indobert_hpo_prediction,
        indobert_hpo_confidence,
        indobert_hpo_class_confidence,
        indobert_hpo_scores
    ) = predict_indobert(
        indobert_hpo,
        tokenizer_hpo,
        indobert_input,
        indobert_label_map,
        max_length=128
    )


    # =====================================================
    # HASIL PREDIKSI
    # =====================================================

    st.divider()

    st.subheader("Hasil Prediksi")


    # -----------------------------------------------------
    # ROW 1 - SVM
    # -----------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        st.markdown("### SVM Baseline")

        if svm_baseline_prediction == "Pro":

            st.success(
                f"## {svm_baseline_prediction}"
            )

        elif svm_baseline_prediction == "Kontra":

            st.error(
                f"## {svm_baseline_prediction}"
            )

        else:

            st.info(
                f"## {svm_baseline_prediction}"
            )

        st.metric(
            "Confidence Score",
            f"{svm_baseline_confidence:.2f}%"
        )


    with col2:

        st.markdown("### SVM HPO")

        if svm_hpo_prediction == "Pro":

            st.success(
                f"## {svm_hpo_prediction}"
            )

        elif svm_hpo_prediction == "Kontra":

            st.error(
                f"## {svm_hpo_prediction}"
            )

        else:

            st.info(
                f"## {svm_hpo_prediction}"
            )

        st.metric(
            "Confidence Score",
            f"{svm_hpo_confidence:.2f}%"
        )


    # -----------------------------------------------------
    # ROW 2 - INDOBERT
    # -----------------------------------------------------

    col3, col4 = st.columns(2)


    with col3:

        st.markdown("### IndoBERT Baseline")

        if indobert_baseline_prediction == "Pro":

            st.success(
                f"## {indobert_baseline_prediction}"
            )

        elif indobert_baseline_prediction == "Kontra":

            st.error(
                f"## {indobert_baseline_prediction}"
            )

        else:

            st.info(
                f"## {indobert_baseline_prediction}"
            )

        st.metric(
            "Confidence Score",
            f"{indobert_baseline_confidence:.2f}%"
        )


    with col4:

        st.markdown("### IndoBERT HPO")

        if indobert_hpo_prediction == "Pro":

            st.success(
                f"## {indobert_hpo_prediction}"
            )

        elif indobert_hpo_prediction == "Kontra":

            st.error(
                f"## {indobert_hpo_prediction}"
            )

        else:

            st.info(
                f"## {indobert_hpo_prediction}"
            )

        st.metric(
            "Confidence Score",
            f"{indobert_hpo_confidence:.2f}%"
        )


    # =====================================================
    # CONFIDENCE SCORE PER KELAS
    # =====================================================

    st.divider()

    st.subheader(
        "Confidence Score per Kelas"
    )

    st.caption(
        "Pada SVM, confidence score dihitung dari "
        "decision score yang dinormalisasi menggunakan "
        "softmax. Pada IndoBERT, nilai ditampilkan "
        "berdasarkan probabilitas softmax dari logits model."
    )


    # -----------------------------------------------------
    # DATAFRAME CONFIDENCE
    # -----------------------------------------------------

    confidence_df = pd.DataFrame({

        "Kelas": [
            "Pro",
            "Kontra",
            "Netral"
        ],

        "SVM Baseline": [

            svm_baseline_class_confidence.get(
                "Pro",
                0
            ),

            svm_baseline_class_confidence.get(
                "Kontra",
                0
            ),

            svm_baseline_class_confidence.get(
                "Netral",
                0
            )
        ],

        "SVM HPO": [

            svm_hpo_class_confidence.get(
                "Pro",
                0
            ),

            svm_hpo_class_confidence.get(
                "Kontra",
                0
            ),

            svm_hpo_class_confidence.get(
                "Netral",
                0
            )
        ],

        "IndoBERT Baseline": [

            indobert_baseline_class_confidence.get(
                "Pro",
                0
            ),

            indobert_baseline_class_confidence.get(
                "Kontra",
                0
            ),

            indobert_baseline_class_confidence.get(
                "Netral",
                0
            )
        ],

        "IndoBERT HPO": [

            indobert_hpo_class_confidence.get(
                "Pro",
                0
            ),

            indobert_hpo_class_confidence.get(
                "Kontra",
                0
            ),

            indobert_hpo_class_confidence.get(
                "Netral",
                0
            )
        ]
    })


    # -----------------------------------------------------
    # CHART
    # -----------------------------------------------------

    st.bar_chart(
        confidence_df.set_index(
            "Kelas"
        )
    )


    # =====================================================
    # DETAIL CONFIDENCE
    # =====================================================

    st.subheader(
        "Detail Confidence Score"
    )


    # -----------------------------------------------------
    # SVM
    # -----------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            "#### SVM Baseline"
        )

        st.write(
            f"**Pro:** "
            f"{svm_baseline_class_confidence.get('Pro', 0):.2f}%"
        )

        st.write(
            f"**Kontra:** "
            f"{svm_baseline_class_confidence.get('Kontra', 0):.2f}%"
        )

        st.write(
            f"**Netral:** "
            f"{svm_baseline_class_confidence.get('Netral', 0):.2f}%"
        )


    with col2:

        st.markdown(
            "#### SVM HPO"
        )

        st.write(
            f"**Pro:** "
            f"{svm_hpo_class_confidence.get('Pro', 0):.2f}%"
        )

        st.write(
            f"**Kontra:** "
            f"{svm_hpo_class_confidence.get('Kontra', 0):.2f}%"
        )

        st.write(
            f"**Netral:** "
            f"{svm_hpo_class_confidence.get('Netral', 0):.2f}%"
        )


    # -----------------------------------------------------
    # INDOBERT
    # -----------------------------------------------------

    col3, col4 = st.columns(2)


    with col3:

        st.markdown(
            "#### IndoBERT Baseline"
        )

        st.write(
            f"**Pro:** "
            f"{indobert_baseline_class_confidence.get('Pro', 0):.2f}%"
        )

        st.write(
            f"**Kontra:** "
            f"{indobert_baseline_class_confidence.get('Kontra', 0):.2f}%"
        )

        st.write(
            f"**Netral:** "
            f"{indobert_baseline_class_confidence.get('Netral', 0):.2f}%"
        )


    with col4:

        st.markdown(
            "#### IndoBERT HPO"
        )

        st.write(
            f"**Pro:** "
            f"{indobert_hpo_class_confidence.get('Pro', 0):.2f}%"
        )

        st.write(
            f"**Kontra:** "
            f"{indobert_hpo_class_confidence.get('Kontra', 0):.2f}%"
        )

        st.write(
            f"**Netral:** "
            f"{indobert_hpo_class_confidence.get('Netral', 0):.2f}%"
        )


    # =====================================================
    # PERBANDINGAN CONFIDENCE MODEL
    # =====================================================

    st.divider()

    st.subheader(
        "Perbandingan Confidence Model"
    )


    comparison_df = pd.DataFrame({

        "Model": [
            "SVM Baseline",
            "SVM HPO",
            "IndoBERT Baseline",
            "IndoBERT HPO"
        ],

        "Confidence Score": [

            svm_baseline_confidence,

            svm_hpo_confidence,

            indobert_baseline_confidence,

            indobert_hpo_confidence
        ]
    })


    st.bar_chart(
        comparison_df.set_index(
            "Model"
        )
    )


    # =====================================================
    # RINGKASAN HASIL
    # =====================================================

    st.subheader("Ringkasan")


    result_df = pd.DataFrame({

        "Model": [

            "SVM Baseline",
            "SVM HPO",
            "IndoBERT Baseline",
            "IndoBERT HPO"
        ],

        "Prediksi": [

            svm_baseline_prediction,
            svm_hpo_prediction,
            indobert_baseline_prediction,
            indobert_hpo_prediction
        ],

        "Confidence Score": [

            f"{svm_baseline_confidence:.2f}%",
            f"{svm_hpo_confidence:.2f}%",
            f"{indobert_baseline_confidence:.2f}%",
            f"{indobert_hpo_confidence:.2f}%"
        ]
    })


    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )