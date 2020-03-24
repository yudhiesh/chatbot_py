import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json 
import pickle 
import numpy as np
import tensorflow as tf
import random 

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)


# Tokenize each word using nltk 

for intent in intents['intents']:
    # Intent shows the grouping of words by their categories 
    for word in intent['patterns']:
        # Word shows all the responses 
        w = nltk.word_tokenize(word)
        # ['List', 'of', 'pharmacies', 'nearby']
        words.extend(w)
        documents.append((w, intent['tag']))
        # Greetings - Hi there 

        # adding classes to class list 
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lowercase all the words 
# Lemmatize - turn words into their root word
# walked - walk
# Narrows the search for words 
# Similar to stemming 
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort them in order 
classes = sorted(list(set(classes)))

#print(len(documents), "documents")
#print(len(classes), "classes")
#print(len(words), "unique words")
# 47 documents
# 9 classes
# 88 words
pickle.dump(words,open('words.pkl', 'wb'))
pickle.dump(classes,open('classes.pkl', 'wb'))


# Building the deep learning model 
# Set up training data 
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # Set up bag of words 
    # Doc shows Words and tags 
    # Bag of words 
    bag = []
    pattern_words = doc[0]
    # ['List', 'of', 'pharmacies', 'nearby']
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in pattern_words]
    # Use 1 hot encoding 
    # If word found in bag count it as 1 and 0 else
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    #print(bag)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
    
#shufflle the features and turn into np.array
random.shuffle(training)
training = np.array(training)
# Seperate them into train and test data 
train_x = list(training[:,0])
train_y = list(training[:,1])

# Create a model - 3 layers 
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")