# Sistema de Detección de Enfermedades en Plantas (con Teachable Machine)

Aplicación web en Python (Streamlit) donde un agricultor sube la foto
de una planta y el sistema le dice qué enfermedad detecta, usando un
modelo entrenado con **Teachable Machine** (sin escribir código para
entrenarlo).

## Estructura del proyecto

```
plant-disease-system/
├── app.py                # Aplicación web para el agricultor
├── requirements.txt      # Dependencias
├── keras_model.h5        # ⬅ lo agregas tú (exportado de Teachable Machine)
├── labels.txt            # ⬅ lo agregas tú (exportado de Teachable Machine)
└── README.md
```

## Paso 1 — Entrenar el modelo en Teachable Machine

1. Descarga el dataset **PlantVillage** desde Kaggle (botón de descarga,
   sin necesitar API ni código):
   https://www.kaggle.com/datasets/emmarex/plantdisease
2. Elige 4-6 carpetas de enfermedades (recomendado: de una sola planta,
   por ejemplo las que empiezan con `Tomato___`).
3. Entra a https://teachablemachine.withgoogle.com/ → **Comenzar** →
   **Proyecto de imagen** → **Modelo estándar de imagen**.
4. Crea una clase por cada carpeta que elegiste, nómbrala claramente
   (ej: `Tomato_Early_blight`) y arrastra ahí las imágenes de esa carpeta.
5. Haz clic en **Entrenar modelo** (tarda pocos minutos, corre en tu navegador).
6. Prueba el modelo con la vista previa. Si falla mucho en alguna clase,
   agrégale más imágenes y vuelve a entrenar.
7. Haz clic en **Exportar modelo** → pestaña **Tensorflow** → selecciona
   **Keras** → **Descargar mi modelo**.
8. Descomprime el `.zip` descargado: contiene `keras_model.h5` y
   `labels.txt`. Coloca ambos archivos en esta misma carpeta del proyecto.

## Paso 2 — Instalar dependencias

```bash
pip install -r requirements.txt
```

## Paso 3 — Ejecutar la aplicación

```bash
streamlit run app.py
```

Se abrirá una página en tu navegador donde puedes subir una foto de una
planta y ver el diagnóstico con el porcentaje de confianza.

## Notas para tu presentación / informe del curso

- El modelo fue entrenado con Teachable Machine, que internamente usa
  **transfer learning** sobre una red MobileNet ya preentrenada —
  puedes explicar esto en tu informe como la técnica usada.
- Puedes mostrar capturas de la pantalla de entrenamiento y la gráfica
  de precisión que Teachable Machine muestra al terminar.
- Cada vez que un agricultor sube una foto, la app la guarda
  automáticamente en la carpeta `historial/`, junto con un archivo
  `historial/registro.csv` que anota fecha, hora, nombre de la imagen,
  diagnóstico y porcentaje de confianza. Desde la propia app puedes
  abrir "📋 Ver historial de consultas guardadas" para revisarlas.
  Esto te sirve como evidencia de uso real para tu informe.
