# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Using the required libraries 
import json
import pickle
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout


# downloading packages
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')

# Our word lemmatizer
lmtzr = WordNetLemmatizer()

# List of words from json file
words = []

# A list for intents
classes = []

# A list for intents and their patterns
documents = []

# Non-alphabetical characters to be ignored
ignore_letters = [',', '.', '!', '?']

# Opening data used for our chatbot
intents_file = open('dataset.json').read()

# Loading data
intents = json.loads(intents_file)

# Traversing every intent in our json file
# and their patterns (possible inputs)
for intent in intents['intents']:
   
    for pattern in intent['patterns']:
       
        # Tokenizing each word in the json file
        word = nltk.word_tokenize(pattern)
        words.extend(word)

        # Adding documents in the corpus
        documents.append((word, intent['tag']))

        # Adding intents (json) to our classes list
        if intent['tag'] not in classes:
            
            classes.append(intent['tag'])

# Output: all the words from the json file
print(documents)

# Lowering, lemmatizing each word, and removing all duplicates
# that may exist in our json file
words = [lmtzr.lemmatize(wrd.lower()) for wrd in words if wrd not in ignore_letters]
words = sorted(list(set(words)))

# Sorting classes
classes = sorted(list(set(classes)))

# Documents (list): consists of all intents and their patterns
print(len(documents), "documents")

# Classes (list): consists of intents
print(len(classes), "classes", classes)

# Words (list): consists of all the words from the json file
print(len(words), "unique words (lemmatized)", words)

# Storing words data to words.pkl
pickle.dump(words, open('words.pkl', 'wb'))

# Storing classes data to classes.pkl
pickle.dump(classes, open('classes.pkl', 'wb'))

# A list for training data
training = []

# Training set, bag of words for each sentence
for document in documents:
    
    # Initializing our bag of words list
    bag_of_words = []

    # List of tokenized words for the pattern
    pattern_words = document[0]

    # Lemmatizing each word, creating base word, to represent related words
    pattern_words = [lmtzr.lemmatize(word.lower()) for word in pattern_words]

    # Creating our bag of words array with 1 if word match found in current pattern
    for word in words:
       
        bag_of_words.append(1) if word in pattern_words else bag_of_words.append(0)

    # Output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = [0] * len(classes)
    output_row[classes.index(document[1])] = 1

    training.append([bag_of_words, output_row])

# Shuffling features and turning them into np.array
random.shuffle(training)

train_x = np.array([row[0] for row in training])
train_y = np.array([row[1] for row in training])

print("Training data have been created successfully")

# Model creation (Sequential) of 3 layers.
# First layer has 128 neurons,
# Second layer has 64 neurons and
# 3rd output layer contains number of neurons equal to
# number of intents to predict output intent with softmax

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Fitting the model using train_x and train_y
hist = model.fit(train_x, train_y, epochs=900, batch_size=64, verbose=1)

# Saving trained model
model.save('Chatbot_model.keras') 