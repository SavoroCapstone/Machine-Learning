from flask import Flask, jsonify
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
import json

app = Flask(__name__)

@app.route('/prediction/<seed_text>')
def get_predictions(seed_text):
    # Load the tokenizer from JSON file
    with open('tokenizer.json', 'r') as f:
        tokenizer_json = json.load(f)

    tokenizer = tokenizer_from_json(tokenizer_json)

    # Load the model
    model = load_model('spell_predict_version_1.h5')

    # Define total sentences to predict
    total_sentences = 5
    max_sequence_len = 20

    # Initialize a list to store the predicted sentences
    predicted_sentences = []

    # Loop until the desired number of sentences is reached
    for _ in range(total_sentences):
        # Define total words to predict for each sentence
        next_words = np.random.choice([1, 2, 3])  # Randomly select the number of next words to predict

        # Reset seed text for each sentence
        sentence = seed_text

        # Generate sentence
        for _ in range(next_words):
            # Convert the seed text to a token sequence
            token_list = tokenizer.texts_to_sequences([sentence])[0]

            # Pad the sequence
            token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

            # Feed to the model and get the probabilities for each index
            probabilities = model.predict(token_list)

            # Pick a random number from [1,2,3]
            choice = np.random.choice(range(1, 11))

            # Sort the probabilities in ascending order
            # and get the random choice from the end of the array
            predicted = np.argsort(probabilities)[0][-choice]

            # Ignore if index is 0 because that is just the padding.
            if predicted != 0:
                # Look up the word associated with the index.
                output_word = tokenizer.index_word[predicted]

                # Combine with the seed text
                sentence += " " + output_word

        # Add the predicted sentence to the list
        predicted_sentences.append(sentence)

    # Create a dictionary to hold the result
    result = {
        "predictions": predicted_sentences
    }

    # Return the result as JSON response
    return jsonify(result)

if __name__ == '__main__':
    app.run()