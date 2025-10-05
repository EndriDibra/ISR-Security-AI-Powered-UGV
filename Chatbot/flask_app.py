# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# importing required libraries 
import json
import random
import pickle
import numpy as np
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
from flask import Flask, render_template, request, jsonify


# creating the main frame for our app
app = Flask(__name__)


# html template (design) for our app
@app.get("/")
def home():

    return render_template("base.html")


@app.route("/predict", methods=["POST"])
def get_bot_response():
   
    try:
       
        userText = request.get_json().get('msg')
        
        response = ai_chatbot_response(userText)
        
        message = {"answer": response}
        
        return jsonify(message)
    
    except Exception as e:
        
        # logging the error for debugging
        print(f"Error in get_bot_response: {e}")
        
        return jsonify({"error": "An error occurred"})


# our word lemmatizer
lmtzr = WordNetLemmatizer()

# loading our model (Sequential)
model = load_model("Chatbot_model.keras")

# reading and loading our dataset (json file)
intents = json.loads(open("dataset.json").read())

# loading words data
words = pickle.load(open("words.pkl", "rb"))

# loading classes data
classes = pickle.load(open("classes.pkl", "rb"))

# a list to store user's texts for sentiment analysis
texts = []

# processing and organizing properly words
def cleaning_sentence(sentence):

    sentence_words = nltk.word_tokenize(sentence)
    
    sentence_words = [lmtzr.lemmatize(word.lower()) for word in sentence_words]

    # tokenizing sentences
    tokenize_sentence = sentence.split()

    # adding user's texts into the list
    texts.extend(tokenize_sentence)

    return sentence_words


# returning bag of words array: 0 or 1 for words
# that exist in sentence
def bag_of_words(s, words_a, show_details=True):

    # tokenizing patterns
    sentence_words = cleaning_sentence(s)

    # initializing our bag of words list
    bag = [0] * len(words_a)

    for sentence in sentence_words:

        for i, word in enumerate(words_a):

            if word == sentence:

                # assigning 1 if current word is in the vocabulary position
                bag[i] = 1

                if show_details:

                    print("found in bag: %s" % word)

    return np.array(bag)


def predict_fun(sentence, model):

    # filter out predictions below a threshold
    p = bag_of_words(sentence, words, show_details=False)
    
    res = model.predict(np.array([p]))[0]

    ERROR_THRESHOLD = 0.25
    
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # sorting by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []

    for r in results:

        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    return return_list


# getting responses
def get_response(ints, intents_json):

    tag = ints[0]["intent"]
    
    list_of_intents = intents_json["intents"]

    for i in list_of_intents:

        if i["tag"] == tag:

            result = random.choice(i["responses"])

            break

    return result


# initializing chatbot responder
def ai_chatbot_response(msg):

    try:
        
        ints = predict_fun(msg, model)
    
        res = get_response(ints, intents)
        
        return res
    
    except Exception as e:
       
        # Log the error for debugging
        print(f"Error in ai chatbot response: {e}")
        
        return "Sorry, I couldn't understand that."


# running app
if __name__ == "__main__":

    app.run(debug=True) 