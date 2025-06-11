import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib


df = pd.read_csv('data/cleaned_dataset.csv')

print(df.head())
print(f"Total samples: {len(df)}")

# Separate features and labels
X = df['clean_headline'] + ' ' + df['clean_body']  # Combine headline and body
y = df['label']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the combined headline and body using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Logistic Regression classifier
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predict on test set
y_pred = model.predict(X_test_vec)

# Print evaluation report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

import os
os.makedirs('saved_model', exist_ok=True)

import pickle

# Save the model
with open('saved_model/logistic_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the TF-IDF vectorizer
with open('saved_model/tfidf_vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Model and vectorizer saved successfully.")
