import numpy as np 
import pandas as pd 
from ast import literal_eval
import warnings
warnings.filterwarnings('ignore')
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from flask import Flask, jsonify


app = Flask(__name__)

def load_and_get_recommendations(title):
    # Load the preprocessed data
    with open('preprocessed_data.pkl', 'rb') as file:
        df = pickle.load(file)

    df['kalori'] = df['kalori'].astype(str)

    # Converts a collection of raw documents to a matrix of TF-IDF features
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(df['overall'])

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    df = df.reset_index(drop=True)
    titles = df['nama']  # Defining a new variable title
    indices = pd.Series(df.index, index=df['nama'])  # Defining a new dataframe indices

    # Convert a collection of text documents to a matrix of token counts
    count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    count_matrix = count.fit_transform(df['overall'])

    # Compute cosine similarity between samples in X and Y.
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    # Define the function that returns 30 most similar movies based on the cosine similarity score
    def get_recommendations(title):
        idx = indices[title]  # Defining a variable with indices
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:31]  # Taking the 30 most similar movies
        movie_indices = [i[0] for i in sim_scores]
        return titles.iloc[movie_indices]  # returns the title based on movie indices

    # Call the get_recommendations function with the provided title
    recommendations = get_recommendations(title)

    # Convert the pandas Series to a list
    recommendations_list = recommendations.tolist()

    # Return the recommendations as a list
    return recommendations_list

from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/recommendations/<title>')
def recommendations(title):
    # Call the load_and_get_recommendations function with the provided title
    recommendations = load_and_get_recommendations(title)

    # Return the recommendations as JSON
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run()