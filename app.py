from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')
nltk.download('wordnet')
app = Flask(__name__)

model = joblib.load('model/SVM_model.joblib')
vectorizer= joblib.load('model/vectorizer.joblib')

lemmatizer = WordNetLemmatizer()
vocab= CountVectorizer()
english_stopwords = set(stopwords.words('english'))
negation_words = {'not', 'no', 'nor', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't","don't","don","over"}
custom_stopwords = english_stopwords - negation_words
def preprocess(raw_text):
    sentence = re.sub("[^a-zA-Z]", " ", raw_text)
    
    sentence = sentence.lower()

    tokens = sentence.split()
                    
    clean_tokens = [t for t in tokens if t not in custom_stopwords]
    
    clean_tokens = [lemmatizer.lemmatize(word) for word in clean_tokens]
    
    return pd.Series([" ".join(clean_tokens)])



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        test = pd.DataFrame({'sample': [review]})
        X_review = test['sample'].apply(lambda x: preprocess(x)) 
        sentiment=model.predict(vectorizer.transform(X_review[0]))[0]
        if sentiment == 1:
            result = 'Positive 😊'
        else:
            result = 'Negative 😞'
        return render_template('index.html', review=review, result=result)

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")
