import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
import numpy as np

# Download tokenizer if not done already
nltk.download('punkt')

# Load data
df = pd.read_csv("data/cleaned_dataset.csv")
texts = (df["headline"] + " " + df["body"]).astype(str)

# Tokenize
tokenized_texts = [word_tokenize(text) for text in texts]

# Train Word2Vec
model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)

# Average word vectors for each sample
def get_vector(tokens):
    vecs = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

vectors = np.array([get_vector(tokens) for tokens in tokenized_texts])

# Create DataFrame and add label
word2vec_df = pd.DataFrame(vectors)
word2vec_df["label"] = df["label"]

# Save to CSV
word2vec_df.to_csv("data/features_word2vec.csv", index=False)

print("✅ Word2Vec feature extraction complete.")