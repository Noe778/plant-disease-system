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
IMG_SIZE = (224, 224)

HISTORIAL_DIR = "historial"
HISTORIAL_CSV = "historial/registro.csv"


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


def preparar_imagen(imagen_pil):
    imagen = ImageOps.fit(imagen_pil.convert("RGB"), IMG_SIZE, Image.Resampling.LANCZOS)
    arr = np.asarray(imagen, dtype=np.float32)
    arr = (arr / 127.5) - 1
    arr = np.expand_dims(arr, axis=0)
    return arr


def nombre_legible(nombre_clase):
    return nombre_clase.replace("___", " - ").replace("_", " ")


def generar_reporte_excel(df):
    """Genera un archivo Excel con formato (colores, encabezado, anchos de
    columna) a partir del historial, ya que un .csv plano no admite estilo."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Historial")
        libro = writer.book
        hoja = writer.sheets["Historial"]

        formato_encabezado = libro.add_format({
            "bold": True,
            "font_color": "#FFFFFF",
            "bg_color": "#1B4332",
            "border": 1,
            "align": "center",
        })
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

    return buffer.getvalue()


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
