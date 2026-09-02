"""
evaluar_modelo.py
------------------
Evalúa el modelo (keras_model.h5) contra una carpeta de imágenes de
validación con etiqueta conocida, y guarda TODAS las métricas en
metricas_modelo.json (el mismo archivo que lee app.py para mostrar el
panel "Validación del modelo").

Calcula:
  - Accuracy, F1 (macro/ponderado), Precisión, Recall, Cohen's Kappa, MCC, AUC
  - MAE, RMSE, R2  ⬅ pedidos por el asesor
  - Métricas por clase (precisión, recall, f1, soporte)
  - Confianza promedio en aciertos / errores
  - Tiempo de inferencia promedio

CÓMO USARLO
-----------
Organiza tu carpeta de imágenes de validación así (una subcarpeta por
clase, igual que como las subiste a Teachable Machine):

    validacion/
    ├── FusariumTR4/   *.jpg
    ├── Cordana/       *.jpg
    ├── Healthy/       *.jpg
    ├── Pestalotiopsis/*.jpg
    └── Sigatoka/      *.jpg

Luego corre:

    python evaluar_modelo.py --carpeta validacion

Sobre MAE, RMSE y R2 en un modelo de CLASIFICACIÓN:
Estas 3 métricas son propias de modelos de REGRESIÓN (predicen un
número). Como tu asesor las pide igual, aquí se calculan de la forma
que se usa académicamente para "traducirlas" a un clasificador:
comparando, para cada foto evaluada, el vector de probabilidades que
entrega el modelo (softmax) contra el vector "ideal" (1 en la clase
correcta, 0 en las demás — codificación one-hot). Así:
  - MAE  = qué tan lejos está, en promedio, la probabilidad que dio el
           modelo del valor ideal (0 o 1).
  - RMSE = igual, pero castigando más fuerte los errores grandes.
  - R2   = qué tanto mejora el modelo respecto a "adivinar" siempre el
           promedio. Puede salir bajo (o hasta negativo) en clasificación
           — es normal y no significa que el modelo sea malo; para eso
           están accuracy/F1/precisión/recall.
"""

import argparse
import json
import os
import time

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,  # nota: se calcula RMSE manualmente (raíz del MSE) para compatibilidad entre versiones de sklearn
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from app import parchear_modelo_antiguo, preparar_imagen  # reutiliza funciones de app.py

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
SALIDA_JSON = "metricas_modelo.json"
EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png")


def cargar_modelo_y_clases():
    parchear_modelo_antiguo(MODEL_PATH)
    modelo = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        clases = [linea.strip().split(" ", 1)[1] for linea in f if linea.strip()]
    return modelo, clases


def listar_imagenes_por_clase(carpeta, clases):
    rutas, etiquetas = [], []
    for idx, clase in enumerate(clases):
        subcarpeta = os.path.join(carpeta, clase)
        if not os.path.isdir(subcarpeta):
            print(f"⚠️  No se encontró la subcarpeta '{clase}' dentro de {carpeta}, se omite.")
            continue
        for nombre in os.listdir(subcarpeta):
            if nombre.lower().endswith(EXTENSIONES_VALIDAS):
                rutas.append(os.path.join(subcarpeta, nombre))
                etiquetas.append(idx)
    return rutas, etiquetas


def evaluar(carpeta):
    modelo, clases = cargar_modelo_y_clases()
    n_clases = len(clases)

    rutas, y_true = listar_imagenes_por_clase(carpeta, clases)
    if not rutas:
        raise SystemExit(
            f"No se encontraron imágenes en '{carpeta}'. Revisa que tenga una "
            f"subcarpeta por cada clase: {clases}"
        )

    y_true = np.array(y_true)
    y_pred = []
    probs_todas = []
    tiempos_ms = []

    print(f"Evaluando {len(rutas)} imágenes...")
    for ruta in rutas:
        imagen = Image.open(ruta)
        entrada = preparar_imagen(imagen)

        inicio = time.perf_counter()
        probs = modelo.predict(entrada, verbose=0)[0]
        tiempos_ms.append((time.perf_counter() - inicio) * 1000)

        probs_todas.append(probs)
        y_pred.append(int(np.argmax(probs)))

    y_pred = np.array(y_pred)
    probs_todas = np.array(probs_todas)  # shape (n_muestras, n_clases)

    # --- Métricas de clasificación estándar ---
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_ponderado = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    precision_ponderada = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    recall_ponderado = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    try:
        y_true_onehot = np.eye(n_clases)[y_true]
        auc_macro = roc_auc_score(y_true_onehot, probs_todas, average="macro", multi_class="ovr")
    except ValueError:
        auc_macro = None

    # --- MAE, RMSE, R2 (probabilidad del modelo vs codificación one-hot ideal) ---
    y_true_onehot = np.eye(n_clases)[y_true]
    mae = mean_absolute_error(y_true_onehot, probs_todas)
    rmse = mean_squared_error(y_true_onehot, probs_todas) ** 0.5
    r2 = r2_score(y_true_onehot, probs_todas)

    # --- Confianza promedio en aciertos / errores ---
    confianzas = probs_todas[np.arange(len(y_pred)), y_pred]
    aciertos = y_pred == y_true
    conf_aciertos = float(confianzas[aciertos].mean()) if aciertos.any() else None
    conf_errores = float(confianzas[~aciertos].mean()) if (~aciertos).any() else None

    # --- Métricas por clase ---
    metricas_por_clase = {}
    prec_c = precision_score(y_true, y_pred, average=None, zero_division=0, labels=range(n_clases))
    rec_c = recall_score(y_true, y_pred, average=None, zero_division=0, labels=range(n_clases))
    f1_c = f1_score(y_true, y_pred, average=None, zero_division=0, labels=range(n_clases))
    for i, clase in enumerate(clases):
        metricas_por_clase[clase] = {
            "precision": round(float(prec_c[i]), 4),
            "recall": round(float(rec_c[i]), 4),
            "f1_score": round(float(f1_c[i]), 4),
            "soporte": int((y_true == i).sum()),
        }

    resultado = {
        "total_imagenes_evaluadas": len(rutas),
        "imagenes_por_clase": {c: int((y_true == i).sum()) for i, c in enumerate(clases)},
        "accuracy": round(float(accuracy), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_ponderado": round(float(f1_ponderado), 4),
        "precision_macro": round(float(precision_macro), 4),
        "precision_ponderada": round(float(precision_ponderada), 4),
        "recall_macro": round(float(recall_macro), 4),
        "recall_ponderado": round(float(recall_ponderado), 4),
        "cohen_kappa": round(float(kappa), 4),
        "mcc": round(float(mcc), 4),
        "auc_macro": round(float(auc_macro), 4) if auc_macro is not None else None,
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
        "mae_rmse_r2_nota": (
            "Este modelo es de CLASIFICACION (no de regresion). MAE/RMSE/R2 "
            "se calcularon comparando el vector de probabilidades (softmax) "
            "de cada imagen evaluada contra su codificacion one-hot ideal. "
            "Valor EXACTO calculado sobre las imagenes de esta evaluacion."
        ),
        "confianza_promedio_aciertos": round(conf_aciertos, 4) if conf_aciertos is not None else None,
        "confianza_promedio_errores": round(conf_errores, 4) if conf_errores is not None else None,
        "tiempo_inferencia_promedio_ms": round(float(np.mean(tiempos_ms)), 2),
        "metricas_por_clase": metricas_por_clase,
    }

    with open(SALIDA_JSON, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Listo. Métricas guardadas en {SALIDA_JSON}")
    print(f"   Accuracy: {accuracy:.4f}   MAE: {mae:.4f}   RMSE: {rmse:.4f}   R2: {r2:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalúa el modelo y guarda sus métricas (incluye MAE/RMSE/R2).")
    parser.add_argument(
        "--carpeta", default="validacion",
        help="Carpeta con subcarpetas por clase (por defecto: 'validacion').",
    )
    args = parser.parse_args()
    evaluar(args.carpeta)
