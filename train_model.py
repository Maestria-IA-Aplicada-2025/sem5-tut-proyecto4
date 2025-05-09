
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import pandas as pd

# Datos de ejemplo (mensaje y su etiqueta correspondiente)
data = {
    'message': ['free money', 'hey, how are you?', 'limited offer', 'let's catch up soon', 'spam message', 'good morning'],
    'label': [1, 0, 1, 0, 1, 0]
}

# Convertir los datos en un DataFrame
df = pd.DataFrame(data)
X = df['message']
y = df['label']

# Ajustar el vectorizador a los datos de entrenamiento
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)  # Ajuste del vectorizador

# Entrenar el modelo
model = MultinomialNB()
model.fit(X_vec, y)

# Guardar el modelo y el vectorizador ajustado
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

print("Modelo y vectorizador entrenados y guardados correctamente.")
