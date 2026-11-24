# =========================================================
# HASIL.PY
# VISUALISASI HASIL PERBANDINGAN MODEL
# =========================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================================================
# KONFIGURASI
# =========================================================

MODELS = [
    "IndoBERT (Base)",
    "IndoBERT (HPO)",
    "SVM (Base)",
    "SVM (HPO)"
]

LABELS = [
    "Pro",
    "Kontra",
    "Netral"
]


# =========================================================
# WARNA MODEL
# =========================================================

MODEL_COLORS = {
    "IndoBERT (Base)": "#2F5BEA",
    "IndoBERT (HPO)": "#7D9CEB",
    "SVM (Base)": "#F2A02F",
    "SVM (HPO)": "#F8D45A"
}


# =========================================================
# HASIL EVALUASI
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
# DATAFRAME HASIL UTAMA
# =========================================================

df_results = pd.DataFrame(RESULTS).T

df_results = df_results[
    [
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted"
    ]
]


# =========================================================
# DATAFRAME HASIL PER KELAS
# =========================================================

class_rows = []

for model in MODELS:

    for label in LABELS:

        class_rows.append({

            "Model": model,

            "Kelas": label,

            "Precision": CLASS_RESULTS[
                model
            ][label]["precision"],

            "Recall": CLASS_RESULTS[
                model
            ][label]["recall"],

            "F1": CLASS_RESULTS[
                model
            ][label]["f1"]
        })


df_class = pd.DataFrame(
    class_rows
)


# =========================================================
# FUNGSI FORMAT PERSENTASE
# =========================================================

def pct(value):
    return f"{value * 100:.2f}%"


# =========================================================
# 1. TABEL HASIL MODEL
# =========================================================

print("\n")
print("=" * 80)
print("HASIL EVALUASI MODEL")
print("=" * 80)

for model in MODELS:

    print(f"\n{model}")

    print(
        f"Accuracy           : "
        f"{pct(RESULTS[model]['accuracy'])}"
    )

    print(
        f"Precision (weighted): "
        f"{pct(RESULTS[model]['precision_weighted'])}"
    )

    print(
        f"Recall (weighted)  : "
        f"{pct(RESULTS[model]['recall_weighted'])}"
    )

    print(
        f"F1 (weighted)      : "
        f"{pct(RESULTS[model]['f1_weighted'])}"
    )


# =========================================================
# 2. PERBANDINGAN 4 MODEL
# =========================================================

metrics = {
    "accuracy": "Accuracy",
    "precision_weighted": "Precision (weighted)",
    "recall_weighted": "Recall (weighted)",
    "f1_weighted": "F1 (weighted)"
}


fig = go.Figure()


for model in MODELS:

    values = [
        RESULTS[model]["accuracy"],
        RESULTS[model]["precision_weighted"],
        RESULTS[model]["recall_weighted"],
        RESULTS[model]["f1_weighted"]
    ]

    fig.add_trace(
        go.Bar(
            name=model,

            x=[
                "Accuracy",
                "Precision (w)",
                "Recall (w)",
                "F1 (weighted)"
            ],

            y=values,

            text=[
                f"{value * 100:.2f}%"
                for value in values
            ],

            textposition="inside",

            marker_color=MODEL_COLORS[model],

            textfont=dict(
                color="white",
                size=13
            )
        )
    )


fig.update_layout(

    title="Perbandingan Kinerja Keempat Model",

    barmode="group",

    yaxis=dict(
        title="Nilai",
        range=[0, 0.80],
        tickformat=".0%"
    ),

    xaxis=dict(
        title=""
    ),

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    ),

    template="plotly_white",

    height=600,

    margin=dict(
        l=60,
        r=40,
        t=100,
        b=80
    )
)

fig.show()


# =========================================================
# 3. F1-SCORE PER KELAS
# =========================================================

fig_class = go.Figure()


for model in MODELS:

    values = [
        CLASS_RESULTS[model][label]["f1"]
        for label in LABELS
    ]

    fig_class.add_trace(
        go.Bar(

            name=model,

            x=LABELS,

            y=values,

            text=[
                f"{value:.3f}"
                for value in values
            ],

            textposition="inside",

            marker_color=MODEL_COLORS[model],

            textfont=dict(
                color="white",
                size=13
            )
        )
    )


fig_class.update_layout(

    title="Perbandingan F1-Score per Kelas",

    barmode="group",

    yaxis=dict(
        title="F1-Score",
        range=[0, 1]
    ),

    xaxis=dict(
        title="Kelas Stance"
    ),

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    ),

    template="plotly_white",

    height=600
)

fig_class.show()


# =========================================================
# 4. PERBANDINGAN INDoBERT BASE VS HPO
# =========================================================

indobert_base = RESULTS[
    "IndoBERT (Base)"
]

indobert_hpo = RESULTS[
    "IndoBERT (HPO)"
]


indobert_metrics = [
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted"
]


indobert_labels = [
    "Accuracy",
    "Precision (w)",
    "Recall (w)",
    "F1 (weighted)"
]


fig_indobert = go.Figure()


fig_indobert.add_trace(
    go.Bar(

        name="Baseline",

        x=indobert_labels,

        y=[
            indobert_base[m]
            for m in indobert_metrics
        ],

        text=[
            f"{indobert_base[m]:.4f}"
            for m in indobert_metrics
        ],

        textposition="inside",

        marker_color="#527CE8",

        textfont=dict(
            color="white"
        )
    )
)


fig_indobert.add_trace(
    go.Bar(

        name="HPO",

        x=indobert_labels,

        y=[
            indobert_hpo[m]
            for m in indobert_metrics
        ],

        text=[
            f"{indobert_hpo[m]:.4f}"
            for m in indobert_metrics
        ],

        textposition="inside",

        marker_color="#A083E6",

        textfont=dict(
            color="white"
        )
    )
)


fig_indobert.update_layout(

    title="Perbandingan IndoBERT: Baseline vs HPO",

    barmode="group",

    yaxis=dict(
        title="Nilai",
        range=[0, 0.80]
    ),

    template="plotly_white",

    height=550
)

fig_indobert.show()


# =========================================================
# 5. PERBANDINGAN SVM BASE VS HPO
# =========================================================

svm_base = RESULTS[
    "SVM (Base)"
]

svm_hpo = RESULTS[
    "SVM (HPO)"
]


svm_metrics = [
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted"
]


svm_labels = [
    "Accuracy",
    "Precision (w)",
    "Recall (w)",
    "F1 (weighted)"
]


fig_svm = go.Figure()


fig_svm.add_trace(
    go.Bar(

        name="Baseline",

        x=svm_labels,

        y=[
            svm_base[m]
            for m in svm_metrics
        ],

        text=[
            f"{svm_base[m]:.4f}"
            for m in svm_metrics
        ],

        textposition="inside",

        marker_color="#527CE8",

        textfont=dict(
            color="white"
        )
    )
)


fig_svm.add_trace(
    go.Bar(

        name="HPO",

        x=svm_labels,

        y=[
            svm_hpo[m]
            for m in svm_metrics
        ],

        text=[
            f"{svm_hpo[m]:.4f}"
            for m in svm_metrics
        ],

        textposition="inside",

        marker_color="#A083E6",

        textfont=dict(
            color="white"
        )
    )
)


fig_svm.update_layout(

    title="Perbandingan SVM: Baseline vs HPO",

    barmode="group",

    yaxis=dict(
        title="Nilai",
        range=[0, 0.80]
    ),

    template="plotly_white",

    height=550
)

fig_svm.show()


# =========================================================
# 6. PERBANDINGAN F1 PER KELAS
#    BASE VS HPO
# =========================================================

fig_f1_models = make_subplots(

    rows=1,

    cols=2,

    subplot_titles=[
        "IndoBERT",
        "SVM"
    ]
)


# ---------------------------------------------------------
# IndoBERT
# ---------------------------------------------------------

fig_f1_models.add_trace(

    go.Bar(

        name="Base",

        x=LABELS,

        y=[
            CLASS_RESULTS[
                "IndoBERT (Base)"
            ][label]["f1"]

            for label in LABELS
        ],

        text=[
            f"{CLASS_RESULTS['IndoBERT (Base)'][label]['f1']:.3f}"

            for label in LABELS
        ],

        textposition="inside",

        marker_color="#527CE8",

        textfont=dict(
            color="white"
        )
    ),

    row=1,

    col=1
)


fig_f1_models.add_trace(

    go.Bar(

        name="HPO",

        x=LABELS,

        y=[
            CLASS_RESULTS[
                "IndoBERT (HPO)"
            ][label]["f1"]

            for label in LABELS
        ],

        text=[
            f"{CLASS_RESULTS['IndoBERT (HPO)'][label]['f1']:.3f}"

            for label in LABELS
        ],

        textposition="inside",

        marker_color="#A083E6",

        textfont=dict(
            color="white"
        )
    ),

    row=1,

    col=1
)


# ---------------------------------------------------------
# SVM
# ---------------------------------------------------------

fig_f1_models.add_trace(

    go.Bar(

        name="Base",

        x=LABELS,

        y=[
            CLASS_RESULTS[
                "SVM (Base)"
            ][label]["f1"]

            for label in LABELS
        ],

        text=[
            f"{CLASS_RESULTS['SVM (Base)'][label]['f1']:.3f}"

            for label in LABELS
        ],

        textposition="inside",

        marker_color="#527CE8",

        textfont=dict(
            color="white"
        ),

        showlegend=False
    ),

    row=1,

    col=2
)


fig_f1_models.add_trace(

    go.Bar(

        name="HPO",

        x=LABELS,

        y=[
            CLASS_RESULTS[
                "SVM (HPO)"
            ][label]["f1"]

            for label in LABELS
        ],

        text=[
            f"{CLASS_RESULTS['SVM (HPO)'][label]['f1']:.3f}"

            for label in LABELS
        ],

        textposition="inside",

        marker_color="#A083E6",

        textfont=dict(
            color="white"
        ),

        showlegend=False
    ),

    row=1,

    col=2
)


fig_f1_models.update_layout(

    title="Perbandingan F1-Score per Kelas: Base vs HPO",

    barmode="group",

    template="plotly_white",

    height=550
)

fig_f1_models.update_yaxes(
    range=[0, 1],
    title="F1-Score"
)

fig_f1_models.show()


# =========================================================
# 7. PERHITUNGAN GAP INDoBERT VS SVM
# =========================================================

gap_base = (

    RESULTS[
        "IndoBERT (Base)"
    ]["f1_weighted"]

    -

    RESULTS[
        "SVM (Base)"
    ]["f1_weighted"]
)


gap_hpo = (

    RESULTS[
        "IndoBERT (HPO)"
    ]["f1_weighted"]

    -

    RESULTS[
        "SVM (HPO)"
    ]["f1_weighted"]
)


# =========================================================
# 8. PERUBAHAN BASE → HPO
# =========================================================

indobert_change = (

    RESULTS[
        "IndoBERT (HPO)"
    ]["f1_weighted"]

    -

    RESULTS[
        "IndoBERT (Base)"
    ]["f1_weighted"]
)


svm_change = (

    RESULTS[
        "SVM (HPO)"
    ]["f1_weighted"]

    -

    RESULTS[
        "SVM (Base)"
    ]["f1_weighted"]
)


print("\n")
print("=" * 80)
print("ANALISIS PERBANDINGAN MODEL")
print("=" * 80)

print(
    f"\nGap F1 weighted IndoBERT vs SVM (Base): "
    f"{gap_base * 100:+.2f} poin persentase"
)

print(
    f"Gap F1 weighted IndoBERT vs SVM (HPO): "
    f"{gap_hpo * 100:+.2f} poin persentase"
)

print(
    f"\nPerubahan IndoBERT setelah HPO: "
    f"{indobert_change * 100:+.2f} poin persentase"
)

print(
    f"Perubahan SVM setelah HPO: "
    f"{svm_change * 100:+.2f} poin persentase"
)


# =========================================================
# 9. MODEL TERBAIK
# =========================================================

best_model = max(

    RESULTS,

    key=lambda model:
        RESULTS[model]["f1_weighted"]
)


best_f1 = RESULTS[
    best_model
]["f1_weighted"]


print(
    f"\nModel dengan F1 weighted tertinggi: "
    f"{best_model}"
)

print(
    f"F1 weighted: "
    f"{best_f1 * 100:.2f}%"
)


# =========================================================
# 10. MODEL TERBAIK PER METRIK
# =========================================================

metric_names = {

    "accuracy":
        "Accuracy",

    "precision_weighted":
        "Precision (weighted)",

    "recall_weighted":
        "Recall (weighted)",

    "f1_weighted":
        "F1 (weighted)"
}


print("\n")
print("=" * 80)
print("MODEL TERBAIK PER METRIK")
print("=" * 80)


for metric, display_name in metric_names.items():

    best = max(

        RESULTS,

        key=lambda model:
            RESULTS[model][metric]
    )

    value = RESULTS[
        best
    ][metric]


    print(
        f"{display_name:<25} "
        f": {best:<20} "
        f"({value * 100:.2f}%)"
    )


# =========================================================
# 11. KESIMPULAN OTOMATIS
# =========================================================

print("\n")
print("=" * 80)
print("KESIMPULAN")
print("=" * 80)


if gap_base > 0:

    print(
        "\nIndoBERT menunjukkan F1-Score weighted "
        "yang lebih tinggi dibandingkan SVM "
        "pada skema Base."
    )

else:

    print(
        "\nSVM menunjukkan F1-Score weighted "
        "yang lebih tinggi dibandingkan IndoBERT "
        "pada skema Base."
    )


if gap_hpo > 0:

    print(
        "Pada skema HPO, IndoBERT juga memiliki "
        "F1-Score weighted yang lebih tinggi "
        "dibandingkan SVM."
    )

else:

    print(
        "Pada skema HPO, SVM memiliki F1-Score "
        "weighted yang lebih tinggi dibandingkan "
        "IndoBERT."
    )


if indobert_change > 0:

    print(
        f"Optimasi hyperparameter meningkatkan "
        f"F1-Score weighted IndoBERT sebesar "
        f"{indobert_change * 100:.2f} poin persentase."
    )

elif indobert_change < 0:

    print(
        f"Optimasi hyperparameter menurunkan "
        f"F1-Score weighted IndoBERT sebesar "
        f"{abs(indobert_change) * 100:.2f} poin persentase."
    )

else:

    print(
        "Optimasi hyperparameter tidak mengubah "
        "F1-Score weighted IndoBERT."
    )


if svm_change > 0:

    print(
        f"Optimasi hyperparameter meningkatkan "
        f"F1-Score weighted SVM sebesar "
        f"{svm_change * 100:.2f} poin persentase."
    )

elif svm_change < 0:

    print(
        f"Optimasi hyperparameter menurunkan "
        f"F1-Score weighted SVM sebesar "
        f"{abs(svm_change) * 100:.2f} poin persentase."
    )

else:

    print(
        "Optimasi hyperparameter tidak mengubah "
        "F1-Score weighted SVM."
    )


print(
    f"Model dengan F1-Score weighted tertinggi "
    f"adalah {best_model} dengan nilai "
    f"{best_f1 * 100:.2f}%."
)


# =========================================================
# 12. TABEL DETAIL PER KELAS
# =========================================================

print("\n")
print("=" * 80)
print("HASIL PER KELAS")
print("=" * 80)


for model in MODELS:

    print(f"\n{model}")

    for label in LABELS:

        result = CLASS_RESULTS[
            model
        ][label]

        print(

            f"{label:<10} "

            f"Precision: "
            f"{result['precision']:.4f} | "

            f"Recall: "
            f"{result['recall']:.4f} | "

            f"F1: "
            f"{result['f1']:.4f}"
        )


# =========================================================
# 13. EXPORT DATA HASIL KE CSV
# =========================================================

df_export = df_results.copy()

df_export.index.name = "Model"

df_export.to_csv(
    "hasil_evaluasi_model.csv"
)


df_class.to_csv(
    "hasil_evaluasi_per_kelas.csv",
    index=False
)


print("\n")
print("File berhasil dibuat:")
print("- hasil_evaluasi_model.csv")
print("- hasil_evaluasi_per_kelas.csv")