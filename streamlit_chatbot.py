# streamlit_healthcare_chatbot_fast.py

import streamlit as st
import pickle
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import os

# ===========================
# Load NLTK only once
# ===========================
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_path)

for pkg in ['punkt', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'tokenizers/{pkg}')  # check if already present
    except LookupError:
        nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()

# ===========================
# Caching Model and Data
# ===========================
@st.cache_resource  # prevents reloading every time Streamlit reruns
def load_resources():
    model = load_model("chatbot_model.h5")
    words = pickle.load(open("words.pkl", "rb"))
    classes = pickle.load(open("classes.pkl", "rb"))
    intents = json.loads(open("intents.json").read())
    return model, words, classes, intents

model, words, classes, intents = load_resources()

# ===========================
# Helper functions
# ===========================
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]  # turn off logs
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if len(intents_list)==0:
        return "Sorry, I couldn't find any matching disease for those symptoms."
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return np.random.choice(i['responses'])

def get_bot_response(user_input):
    prediction = predict_class(user_input)
    response = get_response(prediction, intents)
    return response

# ===========================
# Streamlit App
# ===========================
st.set_page_config(page_title="Healthcare Chatbot", page_icon="🩺")

st.title("🩺 Moon - Healthcare Chatbot")
st.write("Enter your symptoms below and get a possible condition. Type clearly for better results.")

user_input = st.text_input("Your Symptoms:")


if st.button("Get Diagnosis"):
    if user_input.strip() != "":
        response = get_bot_response(user_input)
        st.markdown(f"**Your Symptoms:** {user_input}")
        st.markdown(f"**Possible Condition:** {response}")
    else:
        st.warning("Please enter your symptoms first.")
