# Cleans up sentences that are inputted 

def clean_up_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmetizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Return bag of words array 
# Either 0 or 1 
# Takes the sentences that are cleanes 
# Creates a bag of words that are use for predicting classes 
def bag_of_words(sentence, words, show_details = True):
    # Tokkenize the words 
    sentence_words = clean_up_sentences(sentence)
    # Bag og words 
    # Matrix of N words 
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("Found in bag: %s" % w)
    return(np.array(bag))
# Predict class
# Error threshold of 0.25 to avoid too much overfitting 
# Outputs a list of intents and probabilities
# Which is the likelihood of them matching 

def predict_class(sentence, model):
    # Filter out predictions below threshold 
    p = bag_of_words(sentence, words, show_details = False)
    result = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(result) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Takes the outputted list 
# Checks the json file 
# Outputs the response with the highest probability 

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['response'])
            break
    return result 

# Takes in a msg 
# Predicts the class 
# Puts the output list into getResponse()
# Outputs response 
def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intens)
    return res
