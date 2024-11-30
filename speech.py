import numpy as np
import pickle
import json
from googletrans import Translator
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from gtts import gTTS
import os

lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.model')

intents_json = json.load(open('C:/Users/divij/OneDrive/Desktop/chatbot mini Project/intents.json', encoding='utf-8'))

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

translator = Translator()
def speak(text):
    # Create a gTTS object and save the audio file
    tts = gTTS(text=text, lang='hi')
    tts.save('audio.mp3')
    # Play the audio file using the OS default media player
    os.system('start audio.mp3')

def clean_up_sentence(sentence):
    sentence_words = word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = np.random.choice(i['responses'])
            en_response = result
            hi_response = translator.translate(result, dest='hi').text
            break
    return en_response, hi_response

# GUI code

import tkinter
from tkinter import *

def send_message(message):
    chat_log.config(state=NORMAL)
    chat_log.insert(END, "You: " + message + "\n\n")
    chat_log.config(foreground="#442265", font=("Verdana", 12 ))
    message = bag_of_words(message)
    res = model.predict(np.array([message]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    intents_list = []
    for r in results:
        intents_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    if intents_list:
        en_response, hi_response = get_response(intents_list, intents_json)
        speak(hi_response) # Speak the Hindi response
    else:
        en_response, hi_response = "I'm sorry, I didn't understand that.", "माफ़ कीजिये, मैं समझ नहीं पाया।"
        speak(hi_response) # Speak the Hindi response
    chat_log.insert(END, "Bot: " + en_response + "\n\n")
    chat_log.insert(END, "Bot (in Hindi): " + hi_response + "\n\n")
    chat_log.config(state=DISABLED)
    chat_log.yview(END)

root = Tk()
root.title("Chatbot")

chat_log = Text(root, bd=0, bg="#F0F0F0", height="8", width="50", font="Arial",)

chat_log.config(state=DISABLED)

input_field = Entry(root, bd=0, bg="#F0F0F0", width="29", font="Arial")
input_field.bind("<Return>", (lambda event: send_message(input_field.get())))

send_button = Button(root, text="Send", width="12", height=5,
                     bd=0, bg="#0080ff", activebackground="#00bfff",
                     foreground='#ffffff', font=("Arial", 12), command=lambda: send_message(input_field.get()))

#Place all components on the screen
chat_log.place(x=6, y=6, height=386, width=370)
input_field.place(x=128, y=401, height=40, width=265)
send_button.place(x=6, y=401, height=40)

root.geometry("385x450")
root.mainloop()
