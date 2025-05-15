import pandas as pd
import re

# Load the dataset
df = pd.read_csv('data/dataset.csv')

def clean_bangla_text(text):
    # Remove punctuations, digits, extra spaces
    text = str(text)
    text = re.sub(r'[^\u0980-\u09FF\s]', '', text)  # Keep only Bangla chars and space
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

# Apply cleaning to headline and body
df['clean_headline'] = df['headline'].apply(clean_bangla_text)
df['clean_body'] = df['body'].apply(clean_bangla_text)

# Save cleaned dataset
df.to_csv('data/cleaned_dataset.csv', index=False)

print("✅ Preprocessing complete. Cleaned file saved as data/cleaned_dataset.csv")