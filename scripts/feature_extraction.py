import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

# Create output directory if it doesn't exist
os.makedirs("data/features", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("data/cleaned_dataset.csv")

# TF-IDF Vectorizer for headlines and body text
tfidf_headline = TfidfVectorizer()
tfidf_body = TfidfVectorizer()

# Fit and transform the headline and body text
X_headline = tfidf_headline.fit_transform(df["clean_headline"])
X_body = tfidf_body.fit_transform(df["clean_body"])

# Save the feature matrices
with open("data/features/headline_tfidf.pkl", "wb") as f:
    pickle.dump(X_headline, f)

with open("data/features/body_tfidf.pkl", "wb") as f:
    pickle.dump(X_body, f)

# Save the vectorizers too (for later use in modeling or prediction)
with open("data/features/headline_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_headline, f)

with open("data/features/body_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_body, f)

print("✅ TF-IDF feature extraction complete.")