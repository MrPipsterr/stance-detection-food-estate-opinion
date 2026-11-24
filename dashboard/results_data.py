# =========================================================
# results_data.py
# Data hasil evaluasi model
# =========================================================


# =========================================================
# HASIL METRIK KESELURUHAN
# =========================================================

RESULTS = {

    "IndoBERT (Base)": {
        "accuracy": 0.7119,
        "precision_weighted": 0.7012,
        "recall_weighted": 0.7119,
        "f1_weighted": 0.7027
    },

    "IndoBERT (HPO)": {
        "accuracy": 0.7139,
        "precision_weighted": 0.7035,
        "recall_weighted": 0.7139,
        "f1_weighted": 0.7064
    },

    "SVM (Base)": {
        "accuracy": 0.7208,
        "precision_weighted": 0.6535,
        "recall_weighted": 0.6621,
        "f1_weighted": 0.6566
    },

    "SVM (HPO)": {
        "accuracy": 0.6901,
        "precision_weighted": 0.6853,
        "recall_weighted": 0.6901,
        "f1_weighted": 0.6872
    }
}


# =========================================================
# HASIL PER KELAS
# =========================================================

CLASS_RESULTS = {

    "IndoBERT (Base)": {

        "Pro": {
            "precision": 0.7935,
            "recall": 0.8400,
            "f1": 0.8161
        },

        "Kontra": {
            "precision": 0.6995,
            "recall": 0.7801,
            "f1": 0.7376
        },

        "Netral": {
            "precision": 0.5668,
            "recall": 0.4190,
            "f1": 0.4818
        }
    },

    "IndoBERT (HPO)": {

        "Pro": {
            "precision": 0.8000,
            "recall": 0.8533,
            "f1": 0.8258
        },

        "Kontra": {
            "precision": 0.7090,
            "recall": 0.7592,
            "f1": 0.7332
        },

        "Netral": {
            "precision": 0.5522,
            "recall": 0.4387,
            "f1": 0.4890
        }
    },

    "SVM (Base)": {

        "Pro": {
            "precision": 0.7500,
            "recall": 0.7700,
            "f1": 0.7599
        },

        "Kontra": {
            "precision": 0.6758,
            "recall": 0.7288,
            "f1": 0.7013
        },

        "Netral": {
            "precision": 0.4765,
            "recall": 0.4010,
            "f1": 0.4355
        }
    },

    "SVM (HPO)": {

        "Pro": {
            "precision": 0.7647,
            "recall": 0.7973,
            "f1": 0.7807
        },

        "Kontra": {
            "precision": 0.7069,
            "recall": 0.7199,
            "f1": 0.7134
        },

        "Netral": {
            "precision": 0.5348,
            "recall": 0.4862,
            "f1": 0.5093
        }
    }
}


# =========================================================
# LABEL
# =========================================================

LABELS = ["Pro", "Kontra", "Netral"]


# =========================================================
# KONFIGURASI MODEL
# =========================================================

MODEL_CONFIG = {

    "IndoBERT (Base)": {
        "Model": "indobenchmark/indobert-base-p1",
        "Max Length": "128",
        "Learning Rate": "2e-5",
        "Batch Size": "16",
        "Epochs": "2",
        "Evaluasi": "Stratified 5-Fold"
    },

    "IndoBERT (HPO)": {
        "Model": "indobenchmark/indobert-base-p1",
        "Max Length": "64",
        "Learning Rate": "2e-05",
        "Batch Size": "8",
        "Epochs": "2",
        "Evaluasi": "Stratified 5-Fold"
    },

    "SVM (Base)": {
        "ngram_range" : "(1, 2)",
        "max_features" : "5000",
        "min_df" : "2",
        "max_df" : "0.95",
        "sublinear_tf" : "True",
        "k" : "2000",
        "C" : "1",
        "class_weight" : "balanced",
        "Evaluasi": "Stratified 5-Fold"
    },

    "SVM (HPO)": {
        "ngram_range" : "(1, 1)",
        "max_features" : "3000",
        "min_df" : "1",
        "max_df" : "0.9",
        "sublinear_tf" : "True",
        "k" : "3000",
        "C" : "1",
        "class_weight" : "balanced",
        "Evaluasi": "Stratified 5-Fold"
    }
}