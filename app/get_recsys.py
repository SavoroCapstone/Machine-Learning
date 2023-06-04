from flask import Flask, jsonify
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

app = Flask(__name__)

# Load and preprocess the data
with open('preprocessed_data.pkl', 'rb') as file:
    df = pickle.load(file)
df['nama'] = df['nama'].str.lower()
food_names = df['nama'].values
categories = df['kategori'].values

# Tokenize food names
tokenizer = Tokenizer()
tokenizer.fit_on_texts(food_names)
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(food_names)
padded_sequences = pad_sequences(sequences)

# Load the model and item latent factors
model = keras.models.load_model('recsys_version_1.h5')
item_embedding_model = keras.models.Model(inputs=model.input, outputs=model.layers[0].output)
item_latent_factors = item_embedding_model.predict(padded_sequences)
item_latent_factors = item_latent_factors.reshape((item_latent_factors.shape[0], -1))

# Calculate item-item similarity matrix
item_similarities = np.dot(item_latent_factors, item_latent_factors.T)

@app.route('/recommend/<food_name>')
def recommend_food(food_name):
    # Get recommendations for a specific food
    query_food_index = np.where(food_names == food_name)[0][0]
    query_item_similarities = item_similarities[query_food_index]
    most_similar_indices = np.argsort(query_item_similarities)[-26:-1]  # Get top 25 most similar food indices
    recommended_foods = np.unique(food_names[most_similar_indices])
    
    # Return recommendations as JSON
    return jsonify({'recommendations': recommended_foods.tolist()})

if __name__ == '__main__':
    app.run()
