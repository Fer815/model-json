from flask import Flask, request, jsonify
from tensorflow.keras.models import model_from_json
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# 1. Leer la arquitectura del modelo
with open("Clasi_mascarilla.json", "r") as archivo_json:
    modelo_json = archivo_json.read()

modelo = model_from_json(modelo_json)

# 2. Cargar los pesos
modelo.load_weights("clasi_mascarilla.weights.h5")
print("✅ Modelo cargado correctamente")

# 3. Clases reales del modelo
class_names = ['con_mascarilla', 'mascarilla_mal_puesta', 'sin_mascarilla']

# 4. Función para preprocesar imagen
def procesar_imagen(file_bytes):
    imagen = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    imagen = imagen.resize((150, 150))  # Cambia si tu modelo usa otra dimensión
    imagen = np.array(imagen) / 255.0
    imagen = np.expand_dims(imagen, axis=0)
    return imagen

# 5. Ruta para predicción
@app.route('/predict', methods=['POST'])
def predecir():
    if 'image' not in request.files:
        return jsonify({"error": "No se encontró la imagen"}), 400

    file = request.files['image']
    imagen = procesar_imagen(file.read())
    prediccion = modelo.predict(imagen)
    indice = int(np.argmax(prediccion))
    clase = class_names[indice]
    confianza = float(np.max(prediccion))

    return jsonify({
        "clase": clase,
        "confianza": confianza
    })

# 6. Ejecutar servidor local
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
