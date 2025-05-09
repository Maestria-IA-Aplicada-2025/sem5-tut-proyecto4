
from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Creamos la aplicación Flask
app = Flask(__name__)

# Cargamos el modelo de IA previamente entrenado
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Cargamos el vectorizador para transformar los datos de texto
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Ruta principal para mostrar el formulario
@app.route('/')
def home():
    return render_template('index.html')

# Ruta que maneja la predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Obtenemos el mensaje del formulario
    message = request.form['message']
    
    # Transformamos el mensaje usando el vectorizador
    message_vec = vectorizer.transform([message])
    
    # Realizamos la predicción usando el modelo
    prediction = model.predict(message_vec)
    
    # Si el modelo predice 1, es spam, si predice 0, no es spam
    result = 'Spam' if prediction[0] == 1 else 'No Spam'
    
    # Mostramos el resultado en la página
    return render_template('result.html', prediction=result)

# Iniciamos el servidor Flask
if __name__ == '__main__':
    app.run(debug=True)
