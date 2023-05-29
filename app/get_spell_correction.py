import re
from collections import Counter
import numpy as np
from flask import Flask, jsonify

app = Flask(__name__)

def process_data(file_name):
    words = []
    with open(file_name) as f:
        file_name_data = f.read()
    file_name_data = file_name_data.lower()
    words = re.findall('\w+', file_name_data)
    characters_to_remove = [',', '(', ')', '&', '/', '-', '+', '*', '%', '!']
    words = [word for word in words if word not in characters_to_remove]
    return words

def get_count(word_l):
    word_count_dict = Counter(word_l)
    return word_count_dict

def get_probs(word_count_dict):
    probs = {}
    m = sum(word_count_dict.values())
    for key in word_count_dict.keys():
        probs[key] = word_count_dict[key] / m
    return probs

def delete_letter(word, verbose=False):
    delete_l = []
    split_l = []
    for c in range(len(word)):
        split_l.append((word[:c], word[c:]))
    for a, b in split_l:
        delete_l.append(a + b[1:])
    if verbose:
        print(f"input word {word}, \nsplit_l = {split_l}, \ndelete_l = {delete_l}")
    return delete_l

def switch_letter(word, verbose=False):
    switch_l = []
    split_l = []
    len_word = len(word)
    for c in range(len_word):
        split_l.append((word[:c], word[c:]))
    switch_l = [a + b[1] + b[0] + b[2:] for a, b in split_l if len(b) >= 2]
    if verbose:
        print(f"Input word = {word} \nsplit_l = {split_l} \nswitch_l = {switch_l}")
    return switch_l

def replace_letter(word, verbose=False):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    replace_l = []
    split_l = []
    for c in range(len(word)):
        split_l.append((word[0:c], word[c:]))
    replace_l = [a + l + (b[1:] if len(b) > 1 else '') for a, b in split_l if b for l in letters]
    replace_set = set(replace_l)
    replace_set.remove(word)
    replace_l = sorted(list(replace_set))
    if verbose:
        print(f"Input word = {word} \nsplit_l = {split_l} \nreplace_l {replace_l}")
    return replace_l

def insert_letter(word, verbose=False):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    insert_l = []
    split_l = []
    for c in range(len(word) + 1):
        split_l.append((word[0:c], word[c:]))
    insert_l = [a + l + b for a, b in split_l for l in letters]
    if verbose:
        print(f"Input word {word} \nsplit_l = {split_l} \ninsert_l = {insert_l}")
    return insert_l

def edit_one_letter(word, allow_switches=True):
    edit_one_set = set()
    edit_one_set.update(delete_letter(word))
    if allow_switches:
        edit_one_set.update(switch_letter(word))
    edit_one_set.update(replace_letter(word))
    edit_one_set.update(insert_letter(word))
    return set(edit_one_set)

def edit_two_letters(word, allow_switches=True):
    edit_two_set = set()
    edit_one = edit_one_letter(word, allow_switches=allow_switches)
    for w in edit_one:
        if w:
            edit_two = edit_one_letter(w, allow_switches=allow_switches)
            edit_two_set.update(edit_two)
    return set(edit_two_set)

def get_corrections(word, probs, vocab, n=2, verbose=False):
    suggestions = []
    n_best = []
    suggestions = list((word in vocab and word) or edit_one_letter(word).intersection(vocab) or edit_two_letters(word).intersection(vocab))
    n_best = [[s, probs[s]] for s in list(reversed(suggestions))]
    if verbose:
        print("entered word = ", word, "\nsuggestions = ", suggestions)
    return n_best

@app.route('/corrections/<word>', methods=['GET'])
def get_corrections_endpoint(word):
    word = word.lower()
    corrections = get_corrections(word, probs, vocab, n=1, verbose=False)
    if len(corrections) > 0:
        best_correction = max(corrections, key=lambda x: x[1])
        best_word = best_correction[0]
        best_probability = best_correction[1]
        response = {
            'input_word': word,
            'best_correction': {
                'word': best_word,
                'probability': best_probability
            }
        }
    else:
        response = {
            'input_word': word,
            'best_correction': None
        }
    return jsonify(response)

if __name__ == '__main__':
    word_l = process_data(r'C:\Users\izzat\ansel\column_data_nama.txt')
    vocab = set(word_l)
    word_count_dict = get_count(word_l)
    probs = get_probs(word_count_dict)
    app.run()
