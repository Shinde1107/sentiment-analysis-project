# Sentiment Analysis Model Training
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
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

with open('model.pkl', 'wb') as f:
    pickle.dump((vectorizer, model), f)
