from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Create Flask app
app = Flask(__name__)

# Sample training data
data = {
    'text': ['I love this product', 'This is amazing', 'I am so happy',
             'I hate this', 'This is terrible', 'I am very sad'],
    'label': [1, 1, 1, 0, 0, 0]
}

df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

model = LogisticRegression()
model.fit(X, y)

# Predict sentiment
@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    input_text = input_data['text']
    input_vector = vectorizer.transform([input_text])
    prediction = model.predict(input_vector)
    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
    return jsonify({'sentiment': sentiment})

# Run server
if __name__ == '__main__':
    app.run(debug=True)
