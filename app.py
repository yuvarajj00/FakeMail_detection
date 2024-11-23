from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load your trained model and vectorizer
model = LogisticRegression()
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

# Assuming you've trained the model and saved it, load them here
model = pickle.load(open('model.pkl', 'rb'))
feature_extraction = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the mail content from the form
        input_mail = [request.form['email']]
        
        # Transform input using vectorizer
        input_data_features = feature_extraction.transform(input_mail)
        
        # Predict the category
        prediction = model.predict(input_data_features)
        
        # Generate response
        result = 'Ham mail' if prediction[0] == 1 else 'Spam mail'
        return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == '__main__':
    app.run(debug=True)
