import random 
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import nltk
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import os

import tkinter as tk

# Create the main window
root = tk.Tk()
root.title("Chatbot")

# Create the chat display
chat_display = tk.Text(root, height=20, width=50)
chat_display.grid(row=0, column=0, padx=10, pady=10)

# Create the input box
entry_box = tk.Entry(root, width=30)
entry_box.grid(row=1, column=0, padx=10, pady=10)

# Define the function to handle sending a message
def send_message(event):
    message = entry_box.get()
    if message.strip() == "":
        return "break"
    entry_box.delete(0, tk.END)
    chat_display.insert(tk.END, "You: " + message + "\n")
    chat_display.yview(tk.END)
    intents_list = predict_intent(message)
    response = get_response(intents_list, intents_json)
    chat_display.insert(tk.END, "Chatbot: " + response + "\n")
    chat_display.yview(tk.END)
    return "break"

# Add the send button
send_button = tk.Button(root, text="Send")
send_button.grid(row=1, column=1, padx=10, pady=10)
send_button.bind("<Button-1>", send_message)

# Load the chatbot model and dictionaries
model = keras.models.load_model('chatbot_model.model')
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
intents_json = json.load(open('C:/Users/divij/OneDrive/Desktop/chatbot mini Project/intents.json'))

# Define a function to preprocess the input text
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Define a function to create a bag of words from the preprocessed input text
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Define a function to predict the intent of the input text
# Define a function to predict the intent of the input text
def predict_intent(message):
    p = bag_of_words(message)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    intents_list = []
    for r in results:
        intents_list.append(classes[r[0]])
    return intents_list

# Define a function to get a response based on the predicted intent
def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm sorry, I don't understand. Can you please rephrase your question?"
    
    tag = intents_list[0]
    intents = intents_json['intents']
    for i in intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


root.mainloop()