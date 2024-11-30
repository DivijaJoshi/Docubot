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

import tkinter as tk
from PIL import Image, ImageTk

# Create the main window
root = tk.Tk()
root.title("Chatbot")

# Set the font and color scheme for the widgets
FONT = ("Helvetica", 12)
BACKGROUND_COLOR = "#f2f2f2"
FOREGROUND_COLOR = "#333333"

# Create the chat display
chat_display = tk.Text(root, height=20, width=50, font=FONT, bg=BACKGROUND_COLOR, fg=FOREGROUND_COLOR)
chat_display.pack(side=tk.LEFT, padx=10, pady=10)

# Add an image to the chat display
image = Image.open(r"C:/Users/divij/OneDrive/Desktop/chatbot mini Project/chatbot.png")
photo = ImageTk.PhotoImage(image)
chat_display.image_create(tk.END, image=photo)

# Create a scrollbar for the chat display
scrollbar = tk.Scrollbar(chat_display)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
chat_display.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=chat_display.yview)

# Create the input box
entry_box = tk.Entry(root, width=30, font=FONT, bg=BACKGROUND_COLOR, fg=FOREGROUND_COLOR)
entry_box.pack(side=tk.LEFT, padx=10, pady=10)

# Define the function to handle sending a message
def send_message(event):
    message = entry_box.get()
    if message.strip() == "":
        return None
    entry_box.delete(0, tk.END)
    chat_display.insert(tk.END, "\nYou: " + message)
    response = get_response(predict_intent(message), intents_json)
    chat_display.insert(tk.END, "\nChatbot: " + response)
    chat_display.yview(tk.END)
    return None


# Add the send button
send_button = tk.Button(root, text="Send", font=FONT, bg=BACKGROUND_COLOR, fg=FOREGROUND_COLOR)
send_button.pack(side=tk.LEFT, padx=10, pady=10)
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

# Define a function to convert the input sentence to a bag of words
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

# Define a function to predict the intent of an input sentence
# Define a function to predict the intent of an input sentence
# Define a function to predict the intent of an input sentence
def predict_intent(sentence):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list



def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result



root.bind("<Return>", send_message)
root.mainloop()
