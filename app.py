"""
app.py
-------
Aplicación web (Streamlit) para que un agricultor suba la foto de una
planta y el sistema le diga qué enfermedad tiene (o si está sana).

USO:
    streamlit run app.py

Requiere que ya existan, en la misma carpeta, los archivos exportados
desde Teachable Machine (Exportar modelo → Tensorflow → Keras):
    - keras_model.h5
    - labels.txt
"""

import os

# Los modelos exportados por Teachable Machine usan el formato de una
# versión antigua de Keras (2.x). Esta línea debe ir ANTES de importar
# tensorflow: hace que TensorFlow use su motor de Keras "legado", que sí
# es compatible con ese formato antiguo (las versiones nuevas de Keras 3
# ya no lo son).
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import csv
import io
import json
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
METRICAS_PATH = "metricas_modelo.json"
IMG_SIZE = (224, 224)

HISTORIAL_DIR = "historial"
HISTORIAL_CSV = "historial/registro.csv"

# Modelo binario opcional (Planta / No_Planta), entrenado por separado en
# Teachable Machine, para filtrar fotos que no son de plantas ANTES de
# mandarlas al modelo de diagnóstico. Si estos archivos no existen todavía
# (no lo has entrenado/agregado), la app sigue funcionando y usa como
# respaldo un filtro simple por color.
FILTRO_MODEL_PATH = "modelo_filtro.h5"
FILTRO_LABELS_PATH = "labels_filtro.txt"
UMBRAL_NO_PLANTA = 60.0  # % mínimo de confianza para bloquear una foto


VERDE_HOJA = "#1B4332"
VERDE_CLARO = "#40916C"
AMARILLO_PLATANO = "#F4C95D"
TIERRA = "#6B4226"
ALERTA = "#E4572E"


def inyectar_estilos():
    st.markdown(
        f"""
        <style>
        .banner-hoja {{
            background: linear-gradient(120deg, {VERDE_HOJA} 0%, {VERDE_CLARO} 100%);
            border-radius: 16px;
            padding: clamp(16px, 5vw, 30px);
            margin-bottom: 18px;
            width: 100%;
            box-sizing: border-box;
        }}
        .banner-hoja h1 {{
            color: #FBF7EE;
            font-size: clamp(1.3rem, 5vw, 2.1rem);
            margin: 0;
            line-height: 1.25;
            word-wrap: break-word;
        }}
        .banner-hoja p {{
            color: {AMARILLO_PLATANO};
            margin: 8px 0 0 0;
            font-size: clamp(0.85rem, 2.8vw, 1.02rem);
        }}
        .tarjeta-resultado {{
            border-radius: 14px;
            padding: clamp(14px, 4vw, 24px);
            margin-top: 8px;
            border-left: 7px solid var(--borde-color, {VERDE_CLARO});
            background-color: #ffffff;
            box-shadow: 0 3px 10px rgba(0,0,0,0.06);
            width: 100%;
            box-sizing: border-box;
        }}
        .tarjeta-resultado h3 {{
            margin: 0 0 4px 0;
            color: {VERDE_HOJA};
            font-size: clamp(1.05rem, 3.5vw, 1.4rem);
            word-wrap: break-word;
        }}
        .tarjeta-resultado p {{
            color: {TIERRA} !important;
            font-size: clamp(0.85rem, 2.8vw, 1rem);
        }}
        div.stButton > button, div.stDownloadButton > button {{
            background-color: {VERDE_HOJA};
            color: #FBF7EE;
            border-radius: 10px;
            border: none;
        }}
        div.stButton > button:hover, div.stDownloadButton > button:hover {{
            background-color: {VERDE_CLARO};
            color: #FBF7EE;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def parchear_modelo_antiguo(path):
    """Los modelos exportados por Teachable Machine se generaron con una
    versión antigua de Keras. Las versiones nuevas de TensorFlow ya no
    reconocen un parámetro interno ('groups') de esa versión antigua.
    Esta función lo elimina directamente del archivo antes de cargarlo,
    para que sea compatible."""
    with h5py.File(path, mode="r+") as f:
        if "model_config" not in f.attrs:
            return
        model_config = f.attrs["model_config"]
        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")
        config = json.loads(model_config)

        def limpiar_capas(layers):
            for layer in layers:
                if layer.get("class_name") == "DepthwiseConv2D":
                    layer.get("config", {}).pop("groups", None)
                sub_layers = layer.get("config", {}).get("layers")
                if sub_layers:
                    limpiar_capas(sub_layers)

        limpiar_capas(config["config"]["layers"])
        f.attrs.modify("model_config", json.dumps(config))


def guardar_en_historial(imagen_pil, resultado, confianza):
    """Guarda la foto subida por el agricultor y anota el diagnóstico
    en un CSV, para tener un historial de todas las consultas."""
    os.makedirs(HISTORIAL_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"{timestamp}.jpg"
    ruta_imagen = os.path.join(HISTORIAL_DIR, nombre_archivo)
    imagen_pil.convert("RGB").save(ruta_imagen)

    existe_csv = os.path.isfile(HISTORIAL_CSV)
    with open(HISTORIAL_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not existe_csv:
            writer.writerow(["fecha_hora", "archivo_imagen", "diagnostico", "confianza_%"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            nombre_archivo,
            resultado,
            f"{confianza:.1f}",
        ])

    return ruta_imagen


@st.cache_resource
def cargar_modelo():
    parchear_modelo_antiguo(MODEL_PATH)
    modelo = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        clases = [linea.strip().split(" ", 1)[1] for linea in f if linea.strip()]
    return modelo, clases


@st.cache_resource
def cargar_modelo_filtro():
    """Carga el modelo binario Planta/No_Planta si ya lo agregaste a la
    carpeta. Si no existe todavía, devuelve (None, None) sin dar error,
    para que la app siga funcionando con el filtro de respaldo por color."""
    if not os.path.isfile(FILTRO_MODEL_PATH) or not os.path.isfile(FILTRO_LABELS_PATH):
        return None, None
    try:
        parchear_modelo_antiguo(FILTRO_MODEL_PATH)
        modelo = tf.keras.models.load_model(FILTRO_MODEL_PATH, compile=False)
        with open(FILTRO_LABELS_PATH, "r", encoding="utf-8") as f:
            clases = [linea.strip().split(" ", 1)[1] for linea in f if linea.strip()]
        return modelo, clases
    except Exception:
        return None, None


def preparar_imagen(imagen_pil):
    imagen = ImageOps.fit(imagen_pil.convert("RGB"), IMG_SIZE, Image.Resampling.LANCZOS)
    arr = np.asarray(imagen, dtype=np.float32)
    arr = (arr / 127.5) - 1
    arr = np.expand_dims(arr, axis=0)
    return arr


def nombre_legible(nombre_clase):
    return nombre_clase.replace("___", " - ").replace("_", " ")


UMBRAL_CONTENIDO_VEGETAL = 8.0  # % mínimo de píxeles "tipo hoja" para aceptar la foto (respaldo)


def analizar_contenido_vegetal(imagen_pil, umbral_pct=UMBRAL_CONTENIDO_VEGETAL):
    """FILTRO DE RESPALDO por color (NO es IA, no requiere entrenar nada):
    calcula qué porcentaje de píxeles de la imagen caen en tonos típicos
    de una hoja (verdes, amarillentos y marrones de zonas necrosadas/secas).
    Se usa solo si todavía no agregaste el modelo binario Planta/No_Planta
    (modelo_filtro.h5). Es menos preciso: por ejemplo puede confundir piel
    humana con hojas, porque comparten tonos cafés/marrones.

    Devuelve (es_planta: bool, porcentaje_vegetal: float).
    """
    imagen_hsv = imagen_pil.convert("RGB").resize((150, 150)).convert("HSV")
    arr = np.asarray(imagen_hsv)
    h = arr[:, :, 0].astype(np.int32)
    s = arr[:, :, 1].astype(np.int32)
    v = arr[:, :, 2].astype(np.int32)

    # PIL representa el matiz (hue) en una escala de 0 a 255 (no 0-360).
    # El rango 14-125 cubre amarillo-verdoso, verde y verde-azulado, que
    # es donde caen tanto las hojas sanas como la mayoría de manchas de
    # enfermedad (amarillas/marrones claras). Se exige saturación y brillo
    # mínimos para descartar blancos, grises y negros (paredes, cielo, ropa).
    mascara_vegetal = (h >= 14) & (h <= 125) & (s >= 30) & (v >= 25)

    porcentaje_vegetal = float(mascara_vegetal.mean()) * 100
    es_planta = porcentaje_vegetal >= umbral_pct
    return es_planta, porcentaje_vegetal


def evaluar_si_es_planta(imagen_pil, modelo_filtro, clases_filtro):
    """Decide si la imagen debe pasar al modelo de diagnóstico.
    Prioriza el modelo binario Planta/No_Planta (más preciso) si ya está
    disponible; si no, cae al filtro de respaldo por color.

    Devuelve (es_planta: bool, detalle: str | None) donde 'detalle' es un
    mensaje explicando por qué se bloqueó la imagen (None si se aceptó).
    """
    if modelo_filtro is not None and clases_filtro is not None:
        entrada = preparar_imagen(imagen_pil)
        predicciones = modelo_filtro.predict(entrada, verbose=0)[0]
        indice = int(np.argmax(predicciones))
        etiqueta = clases_filtro[indice]
        confianza = float(predicciones[indice]) * 100

        es_no_planta = "no" in etiqueta.lower()
        if es_no_planta and confianza >= UMBRAL_NO_PLANTA:
            return False, (
                f"El modelo de filtro clasificó la imagen como "
                f"'{etiqueta}' con {confianza:.0f}% de confianza."
            )
        return True, None

    es_planta, pct = analizar_contenido_vegetal(imagen_pil)
    if not es_planta:
        return False, (
            f"Filtro por color (respaldo): solo se detectó un {pct:.0f}% "
            f"de tonos vegetales. Para un filtro más preciso, entrena y "
            f"agrega el modelo binario 'modelo_filtro.h5'."
        )
    return True, None


def cargar_metricas_modelo():
    """Lee metricas_modelo.json si existe. Devuelve None si no está."""
    if not os.path.isfile(METRICAS_PATH):
        return None
    with open(METRICAS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def generar_reporte_excel(df):
    """Genera un archivo Excel con formato (colores, encabezado, anchos de
    columna). Incluye 3 hojas:
      1) Historial: cada consulta hecha por el agricultor (lo que ya existía).
      2) Metricas del Modelo: accuracy, F1, precision, recall, kappa, mcc,
         MAE, RMSE, R2, etc. — la evaluación general del modelo, para que el
         asesor tenga los indicadores completos y no solo la confianza por foto.
      3) Metricas por Clase: precision/recall/f1/soporte de cada enfermedad.
    """
    buffer = io.BytesIO()
    m = cargar_metricas_modelo()

    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        libro = writer.book

        formato_encabezado = libro.add_format({
            "bold": True,
            "font_color": "#FFFFFF",
            "bg_color": "#1B4332",
            "border": 1,
            "align": "center",
        })

        # --- Hoja 1: Historial de consultas (igual que antes) ---
        df.to_excel(writer, index=False, sheet_name="Historial")
        hoja = writer.sheets["Historial"]
        for col_num, nombre_col in enumerate(df.columns):
            hoja.write(0, col_num, nombre_col, formato_encabezado)
            ancho = max(14, len(str(nombre_col)) + 4)
            hoja.set_column(col_num, col_num, ancho)

        formato_sana = libro.add_format({"bg_color": "#D8F3DC"})
        formato_enferma = libro.add_format({"bg_color": "#FDE8D9"})
        col_diagnostico = list(df.columns).index("diagnostico")
        for fila_idx, valor in enumerate(df["diagnostico"], start=1):
            formato = formato_sana if "healthy" in str(valor).lower() else formato_enferma
            hoja.write(fila_idx, col_diagnostico, valor, formato)

        # --- Hoja 2 y 3: métricas del modelo, si metricas_modelo.json existe ---
        if m is not None:
            filas_generales = [
                ("Total de imágenes evaluadas", m.get("total_imagenes_evaluadas")),
                ("Accuracy", m.get("accuracy")),
                ("F1 (macro)", m.get("f1_macro")),
                ("F1 (ponderado)", m.get("f1_ponderado")),
                ("Precisión (macro)", m.get("precision_macro")),
                ("Precisión (ponderada)", m.get("precision_ponderada")),
                ("Recall (macro)", m.get("recall_macro")),
                ("Recall (ponderado)", m.get("recall_ponderado")),
                ("Cohen's Kappa", m.get("cohen_kappa")),
                ("MCC", m.get("mcc")),
                ("AUC (macro)", m.get("auc_macro")),
                ("MAE", m.get("mae")),
                ("RMSE", m.get("rmse")),
                ("R2", m.get("r2")),
                ("Confianza promedio en aciertos", m.get("confianza_promedio_aciertos")),
                ("Confianza promedio en errores", m.get("confianza_promedio_errores")),
                ("Tiempo de inferencia promedio (ms)", m.get("tiempo_inferencia_promedio_ms")),
            ]
            df_general = pd.DataFrame(filas_generales, columns=["Métrica", "Valor"])
            df_general.to_excel(writer, index=False, sheet_name="Metricas del Modelo")
            hoja_gen = writer.sheets["Metricas del Modelo"]
            for col_num, nombre_col in enumerate(df_general.columns):
                hoja_gen.write(0, col_num, nombre_col, formato_encabezado)
            hoja_gen.set_column(0, 0, 34)
            hoja_gen.set_column(1, 1, 16)

            if m.get("mae_rmse_r2_nota"):
                hoja_gen.write(
                    len(filas_generales) + 2, 0,
                    "Nota MAE/RMSE/R2: " + m["mae_rmse_r2_nota"],
                )

            metricas_clase = m.get("metricas_por_clase")
            if metricas_clase:
                filas_clase = []
                for clase, valores in metricas_clase.items():
                    filas_clase.append({
                        "Clase": clase,
                        "Precisión": valores.get("precision"),
                        "Recall": valores.get("recall"),
                        "F1-score": valores.get("f1_score"),
                        "Soporte (fotos evaluadas)": valores.get("soporte"),
                    })
                df_clase = pd.DataFrame(filas_clase)
                df_clase.to_excel(writer, index=False, sheet_name="Metricas por Clase")
                hoja_clase = writer.sheets["Metricas por Clase"]
                for col_num, nombre_col in enumerate(df_clase.columns):
                    hoja_clase.write(0, col_num, nombre_col, formato_encabezado)
                    ancho = max(16, len(str(nombre_col)) + 4)
                    hoja_clase.set_column(col_num, col_num, ancho)

    return buffer.getvalue()


def mostrar_validacion_modelo():
    """Muestra en la app los indicadores de la evaluación completa del
    modelo (F1, Precisión, Recall, etc.), calculados una sola vez con
    evaluar_modelo.py y guardados en metricas_modelo.json. Estos números
    NO cambian con cada foto: resumen qué tan bien rinde el modelo en
    general, según pruebas con fotos de respuesta conocida."""
    if not os.path.isfile(METRICAS_PATH):
        return

    with open(METRICAS_PATH, "r", encoding="utf-8") as f:
        m = json.load(f)

    with st.expander("📊 Validación del modelo (para tu informe/tesis)", expanded=False):
        st.caption(
            f"Calculado sobre {m['total_imagenes_evaluadas']} imágenes de evaluación "
            f"con evaluar_modelo.py."
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
        c2.metric("F1 (macro)", f"{m['f1_macro']:.3f}")
        c3.metric("Precisión (macro)", f"{m['precision_macro']:.3f}")
        c4.metric("Recall (macro)", f"{m['recall_macro']:.3f}")

        c5, c6, c7 = st.columns(3)
        c5.metric("Cohen's Kappa", f"{m['cohen_kappa']:.3f}")
        c6.metric("MCC", f"{m['mcc']:.3f}")
        if m.get("auc_macro") is not None:
            c7.metric("AUC (macro)", f"{m['auc_macro']:.3f}")

        if all(k in m for k in ("mae", "rmse", "r2")):
            st.write("**MAE, RMSE y R² (solicitados para el informe):**")
            c8, c9, c10 = st.columns(3)
            c8.metric("MAE", f"{m['mae']:.3f}")
            c9.metric("RMSE", f"{m['rmse']:.3f}")
            c10.metric("R²", f"{m['r2']:.3f}")
            st.caption(
                "⚠️ Este es un modelo de clasificación, no de regresión: MAE, "
                "RMSE y R² no son sus métricas nativas (para eso están "
                "accuracy, F1, precisión y recall, arriba). Aquí se calculan "
                "comparando la confianza del modelo contra el resultado ideal "
                "(1 si acertó, 0 si falló), que es como suelen pedirse en "
                "informes académicos cuando exigen estas 3 métricas."
            )

        st.write("**Precisión, Recall y F1 por enfermedad:**")
        filas = []
        for clase, valores in m["metricas_por_clase"].items():
            filas.append({
                "Clase": clase,
                "Precisión": valores["precision"],
                "Recall": valores["recall"],
                "F1-score": valores["f1_score"],
                "Fotos evaluadas": valores["soporte"],
            })
        st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

        if os.path.isfile("matriz_confusion.png"):
            st.image("matriz_confusion.png", caption="Matriz de confusión")
        if os.path.isfile("curvas_roc.png"):
            st.image("curvas_roc.png", caption="Curvas ROC por clase")


def mostrar_historial():
    st.subheader("📋 Historial de consultas")
    if not os.path.isfile(HISTORIAL_CSV):
        st.info("Todavía no hay consultas guardadas. Analiza una foto para empezar tu historial.")
        return

    df = pd.read_csv(HISTORIAL_CSV)

    total = len(df)
    sanas = df["diagnostico"].str.contains("healthy", case=False).sum()
    pct_sanas = (sanas / total * 100) if total else 0
    mas_comun = df["diagnostico"].value_counts().idxmax() if total else "-"

    col1, col2, col3 = st.columns(3)
    col1.metric("Consultas totales", total)
    col2.metric("Plantas sanas", f"{pct_sanas:.0f}%")
    col3.metric("Más frecuente", mas_comun)

    st.write("**Casos por diagnóstico:**")
    conteo = df["diagnostico"].value_counts()
    st.bar_chart(conteo)

    def resaltar_fila(fila):
        color = "#D8F3DC" if "healthy" in str(fila["diagnostico"]).lower() else "#FDE8D9"
        return [f"background-color: {color}"] * len(fila)

    st.dataframe(
        df.style.apply(resaltar_fila, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "⬇️ Descargar CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="historial_diagnosticos.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_b:
        st.download_button(
            "📊 Descargar reporte Excel",
            data=generar_reporte_excel(df),
            file_name="reporte_diagnosticos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


def main():
    st.set_page_config(page_title="Detector de Enfermedades en Plantas", page_icon="🌿")
    inyectar_estilos()
    st.markdown(
        """
        <div class="banner-hoja">
            <h1>🌿 Detector de Enfermedades en Plantas</h1>
            <p>Sube o toma una foto de la hoja y recibe un diagnóstico al instante.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("📱 ¿Cómo abrir esto desde tu celular?"):
        st.write(
            "1. Asegúrate de que tu celular esté conectado a la **misma red WiFi** "
            "que tu computadora.\n"
            "2. En tu computadora, busca en la terminal (donde corriste "
            "`streamlit run app.py`) la línea que dice **'Network URL'** "
            "(algo como `http://192.168.x.x:8501`).\n"
            "3. Escribe esa dirección en el navegador de tu celular.\n\n"
            "Nota: la cámara solo funciona por HTTPS o en tu propia laptop — "
            "para usarla desde el celular, publica la app en línea."
        )

    mostrar_validacion_modelo()

    try:
        modelo, clases = cargar_modelo()
    except FileNotFoundError:
        st.error(
            "No se encontró el modelo entrenado. Coloca en esta misma "
            "carpeta los archivos `keras_model.h5` y `labels.txt` que "
            "descargaste al exportar tu modelo desde Teachable Machine."
        )
        return
    except Exception as e:
        st.error(f"No se pudo cargar el modelo. Detalle técnico: {e}")
        return

    modelo_filtro, clases_filtro = cargar_modelo_filtro()

    modo = st.radio(
        "¿Cómo quieres analizar tu planta?",
        ["📷 Tomar foto con la cámara", "📁 Subir una foto"],
        horizontal=True,
    )

    archivo = None
    if modo == "📷 Tomar foto con la cámara":
        archivo = st.camera_input("Apunta la cámara a la hoja y toma la foto")
    else:
        archivo = st.file_uploader("Foto de la planta", type=["jpg", "jpeg", "png"])

    if archivo is not None:
        imagen = Image.open(archivo)
        st.image(imagen, caption="Imagen analizada", use_container_width=True)

        es_planta, detalle_filtro = evaluar_si_es_planta(imagen, modelo_filtro, clases_filtro)

        if not es_planta:
            st.markdown(
                f"""
                <div class="tarjeta-resultado" style="--borde-color: {ALERTA};">
                    <h3>🚫 Esto no parece ser una hoja de planta</h3>
                    <p>{detalle_filtro} Sube una foto donde la hoja se vea
                    clara, ocupe la mayor parte del encuadre y con buena luz.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if modelo_filtro is None:
                st.caption(
                    "⚠️ Todavía no se detectó 'modelo_filtro.h5' en la carpeta — "
                    "se está usando el filtro de respaldo por color, menos preciso."
                )
        else:
            with st.spinner("Analizando la imagen..."):
                entrada = preparar_imagen(imagen)
                predicciones = modelo.predict(entrada)[0]
                indice = int(np.argmax(predicciones))
                confianza = float(predicciones[indice]) * 100
                resultado = nombre_legible(clases[indice])

            st.subheader("Resultado")
            es_sana = "healthy" in clases[indice].lower()
            color_borde = VERDE_CLARO if es_sana else ALERTA
            icono = "✅" if es_sana else "🍂"
            titulo = "Planta sana" if es_sana else f"Posible enfermedad: {resultado}"
            st.markdown(
                f"""
                <div class="tarjeta-resultado" style="--borde-color: {color_borde};">
                    <h3>{icono} {titulo}</h3>
                    <p>Confianza del modelo: <b>{confianza:.1f}%</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(int(confianza), 100))

            guardar_en_historial(imagen, resultado, confianza)

            with st.expander("Ver todas las probabilidades"):
                orden = np.argsort(predicciones)[::-1]
                for i in orden[:5]:
                    st.write(f"- {nombre_legible(clases[i])}: {predicciones[i]*100:.1f}%")

            st.caption(
                "Este resultado es una ayuda automática y no reemplaza el "
                "diagnóstico de un ingeniero agrónomo."
            )

    st.divider()
    mostrar_historial()


if __name__ == "__main__":
    main()
