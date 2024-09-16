import requests
import json
from flask import Flask, request
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import os
from scipy.sparse import issparse
import threading

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

BOT_ID = ""
BOT_NAME = "Echelon"
API_ROOT = 'https://api.groupme.com/v3/'
POST_URL = "https://api.groupme.com/v3/bots/post"
app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

model_lock = threading.Lock()
current_model = None
current_vectorizer = None

def load_training_data(file_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_dir, file_path)
    training_data = []
    
    try:
        with open(full_path, 'r', encoding='utf-8', errors='replace') as file:
            csv_reader = csv.reader(file)
            next(csv_reader, None)  # Skip the header row
            for row in csv_reader:
                if len(row) >= 2:
                    label, message = row[0], row[1]
                    training_data.append((message, label))
    except FileNotFoundError:
        print(f"Error: The file {full_path} was not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {str(e)}")
    
    return training_data

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(filtered_tokens)

def train_model():
    global current_model, current_vectorizer
    
    training_data = load_training_data('spam.csv')
    if not training_data:
        print("Error: No training data available. Model training aborted.")
        return
    
    X = [preprocess_text(text) for text, _ in training_data]
    y = [label for _, label in training_data]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    new_vectorizer = TfidfVectorizer()
    X_train_tfidf = new_vectorizer.fit_transform(X_train)

    new_model = SVC(kernel='linear', probability=True)
    new_model.fit(X_train_tfidf, y_train)

    with model_lock:
        current_model = new_model
        current_vectorizer = new_vectorizer
    
    print("Model retrained and updated successfully")

def classify_message(message):
    preprocessed_message = preprocess_text(message)
    
    with model_lock:
        if current_model is None or current_vectorizer is None:
            return 0.5  # Return default probability if model is not ready
        
        tfidf_feature_vector = current_vectorizer.transform([preprocessed_message])
        spam_probability = current_model.predict_proba(tfidf_feature_vector)[0][1]
    
    return spam_probability

def update_csv(message, label):
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spam.csv')
    
    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            spam_writer = csv.writer(csvfile)
            spam_writer.writerow([label, message,'', '', ''])
        print(f"Successfully added {label} message to spam.csv")
    except Exception as e:
        print(f"Error adding message to spam.csv: {str(e)}")
    
    # Retrain the model after each new message
    threading.Thread(target=train_model).start()

def handle_message(message, user_id, group_id, message_id, sender_id):
    print(f"Received message: {message}")
    
    try:
        spam_probability = classify_message(message)
        print(f"Spam probability: {spam_probability:.2%}")
        
        # Classify as spam if probability > 0.5, otherwise as ham
        label = 'spam' if spam_probability > 0.5 else 'ham'
        update_csv(message, label)
        
        print(f"Message added to spam.csv as {label}")
    
    except Exception as e:
        print(f"An error occurred while processing the message: {str(e)}")

@app.route('/', methods=['POST'])
def webhook():
    data = request.get_json()
    
    if data['name'] != BOT_NAME:
        message = data['text']
        user_id = data['user_id']
        group_id = data['group_id']
        message_id = data['id']
        sender_id = data['sender_id']
        
        handle_message(message, user_id, group_id, message_id, sender_id)
    
    return "OK", 200

if __name__ == "__main__":
    print(f"Starting data collector bot with BOT_ID: {BOT_ID}")
    train_model()  # Initial model training
    app.run(debug=True, port=5000)