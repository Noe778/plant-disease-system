"""
evaluar_modelo.py
-----------------
Evalúa el modelo con fotos de prueba (que NO se usaron para entrenar) y
genera lo que la app muestra en "Validación del modelo":
    - metricas_modelo.json
    - matriz_confusion.png
    - curvas_roc.png

USO:
    python evaluar_modelo.py [carpeta_pruebas]

La carpeta de pruebas (por defecto "pruebas") debe tener una subcarpeta por
clase, con el MISMO nombre que en labels.txt:

    pruebas/
        Cordana/        *.jpg
        Sigatoka/       *.jpg
        Pestalotiopsis/ *.jpg
        Sanas/          *.jpg
        Otros/          *.jpg

Requiere: tensorflow, tf_keras, h5py, scikit-learn, matplotlib, pillow, numpy.
"""

import json
import os
import sys
import time

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
IMG_SIZE = (224, 224)
EXTENSIONES = (".jpg", ".jpeg", ".png")


def leer_clases(path=LABELS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return [linea.strip().split(" ", 1)[1] for linea in f if linea.strip()]


def parchear_modelo_antiguo(path):
    """Quita el parámetro 'groups' que las versiones nuevas de Keras ya no
    reconocen en los modelos exportados por Teachable Machine (igual que en
    app.py)."""
    import h5py

    with h5py.File(path, mode="r+") as f:
        if "model_config" not in f.attrs:
            return
        model_config = f.attrs["model_config"]
        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")
        config = json.loads(model_config)

        def limpiar(layers):
            for layer in layers:
                if layer.get("class_name") == "DepthwiseConv2D":
                    layer.get("config", {}).pop("groups", None)
                sub = layer.get("config", {}).get("layers")
                if sub:
                    limpiar(sub)

        limpiar(config["config"]["layers"])
        f.attrs.modify("model_config", json.dumps(config))


def preparar_imagen(ruta):
    """Mismo preprocesado que app.py: recorte central a 224x224 y [-1, 1]."""
    from PIL import Image, ImageOps

    imagen = ImageOps.fit(Image.open(ruta).convert("RGB"), IMG_SIZE, Image.Resampling.LANCZOS)
    arr = np.asarray(imagen, dtype=np.float32)
    return (arr / 127.5) - 1


def calcular_metricas(y_true, probs, clases, tiempos_ms=None):
    """Calcula todas las métricas a partir de las etiquetas reales (índices)
    y las probabilidades (n_imagenes x n_clases)."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    y_pred = probs.argmax(axis=1)
    n = len(clases)
    etiquetas = list(range(n))

    prec, rec, f1, sop = precision_recall_fscore_support(
        y_true, y_pred, labels=etiquetas, zero_division=0
    )

    y_bin = label_binarize(y_true, classes=etiquetas)
    try:
        auc_macro = float(roc_auc_score(y_bin, probs, average="macro", multi_class="ovr"))
    except ValueError:
        auc_macro = None

    acierto = y_pred == y_true
    conf = probs.max(axis=1)

    def r(x, d=4):
        return round(float(x), d)

    metricas = {
        "total_imagenes_evaluadas": int(len(y_true)),
        "imagenes_por_clase": {c: int(sop[i]) for i, c in enumerate(clases)},
        "accuracy": r(accuracy_score(y_true, y_pred)),
        "f1_macro": r(f1_score(y_true, y_pred, average="macro", labels=etiquetas, zero_division=0)),
        "f1_ponderado": r(f1_score(y_true, y_pred, average="weighted", labels=etiquetas, zero_division=0)),
        "precision_macro": r(precision_score(y_true, y_pred, average="macro", labels=etiquetas, zero_division=0)),
        "precision_ponderada": r(precision_score(y_true, y_pred, average="weighted", labels=etiquetas, zero_division=0)),
        "recall_macro": r(recall_score(y_true, y_pred, average="macro", labels=etiquetas, zero_division=0)),
        "recall_ponderado": r(recall_score(y_true, y_pred, average="weighted", labels=etiquetas, zero_division=0)),
        "cohen_kappa": r(cohen_kappa_score(y_true, y_pred)),
        "mcc": r(matthews_corrcoef(y_true, y_pred)),
        "auc_macro": None if auc_macro is None else r(auc_macro),
        "confianza_promedio_aciertos": r(conf[acierto].mean()) if acierto.any() else None,
        "confianza_promedio_errores": r(conf[~acierto].mean()) if (~acierto).any() else None,
        "tiempo_inferencia_promedio_ms": None if not tiempos_ms else r(np.mean(tiempos_ms), 1),
        "metricas_por_clase": {
            c: {
                "precision": r(prec[i]),
                "recall": r(rec[i]),
                "f1_score": r(f1[i]),
                "soporte": int(sop[i]),
            }
            for i, c in enumerate(clases)
        },
    }
    return metricas, y_pred, y_bin


def guardar_graficos(y_true, y_pred, probs, y_bin, clases, carpeta="."):
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(clases))))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=clases).plot(
        ax=ax, cmap="Greens", xticks_rotation=30, colorbar=False
    )
    ax.set_title("Matriz de confusión")
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta, "matriz_confusion.png"), dpi=150)
    plt.close(fig)

    # Curvas ROC por clase
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, c in enumerate(clases):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], np.asarray(probs)[:, i])
        auc_i = roc_auc_score(y_bin[:, i], np.asarray(probs)[:, i])
        ax.plot(fpr, tpr, label=f"{c} (AUC = {auc_i:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curvas ROC por clase")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta, "curvas_roc.png"), dpi=150)
    plt.close(fig)


def main():
    carpeta = sys.argv[1] if len(sys.argv) > 1 else "pruebas"
    clases = leer_clases()

    faltan = [c for c in clases if not os.path.isdir(os.path.join(carpeta, c))]
    if faltan:
        sys.exit(f"Faltan carpetas de prueba para estas clases: {faltan} (en '{carpeta}').")

    import tensorflow as tf

    parchear_modelo_antiguo(MODEL_PATH)
    modelo = tf.keras.models.load_model(MODEL_PATH, compile=False)

    y_true, probs, tiempos = [], [], []
    for idx, clase in enumerate(clases):
        ruta_clase = os.path.join(carpeta, clase)
        archivos = sorted(f for f in os.listdir(ruta_clase) if f.lower().endswith(EXTENSIONES))
        print(f"{clase}: {len(archivos)} fotos")
        for nombre in archivos:
            x = np.expand_dims(preparar_imagen(os.path.join(ruta_clase, nombre)), 0)
            t0 = time.perf_counter()
            p = modelo.predict(x, verbose=0)[0]
            tiempos.append((time.perf_counter() - t0) * 1000)
            y_true.append(idx)
            probs.append(p)

    metricas, y_pred, y_bin = calcular_metricas(y_true, probs, clases, tiempos)
    with open("metricas_modelo.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    guardar_graficos(y_true, y_pred, probs, y_bin, clases)

    print(f"\nAccuracy: {metricas['accuracy']*100:.1f}%   F1 macro: {metricas['f1_macro']:.3f}")
    print("Guardados: metricas_modelo.json, matriz_confusion.png, curvas_roc.png")


if __name__ == "__main__":
    main()
