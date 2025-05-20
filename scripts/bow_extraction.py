import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Load the cleaned dataset
df = pd.read_csv("data/cleaned_dataset.csv")

# Combine headline and body (optional, depends on your model input strategy)
texts = df["headline"] + " " + df["body"]

# Initialize CountVectorizer (BoW)
vectorizer = CountVectorizer(max_features=1000)  # Limit to top 1000 features for simplicity
X_bow = vectorizer.fit_transform(texts)

# Convert to DataFrame
bow_df = pd.DataFrame(X_bow.toarray(), columns=vectorizer.get_feature_names_out())

# Add label column back
bow_df["label"] = df["label"]

# Save to CSV
bow_df.to_csv("data/features_bow.csv", index=False)

print("✅ Bag of Words feature extraction complete.")