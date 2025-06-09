import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

# Load the cleaned dataset
df = pd.read_csv('../data/cleaned_dataset.csv')  # Adjust if running from root

# Show basic info
print("Dataset shape:", df.shape)
print("\nColumns:", df.columns)
print("\nFirst few rows:\n", df.head())

# Check missing values
print("\nMissing values:\n", df.isnull().sum())

# Class distribution (assumes 'mismatch' is the label)
if 'mismatch' in df.columns:
    sns.countplot(data=df, x='mismatch')
    plt.title("Class Distribution (Mismatch)")
    plt.xlabel("Mismatch Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# Add text length feature
df['text_length'] = df['text'].astype(str).apply(lambda x: len(x.split()))

# Distribution of text length
sns.histplot(df['text_length'], bins=30, kde=True)
plt.title("Distribution of Text Length")
plt.xlabel("Number of Words")
plt.tight_layout()
plt.show()

# Most common words (Bangla words)
all_words = ' '.join(df['text'].astype(str)).split()
common_words = Counter(all_words).most_common(20)
words, counts = zip(*common_words)

plt.figure(figsize=(10, 5))
sns.barplot(x=list(words), y=list(counts))
plt.title("Top 20 Most Frequent Words")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# WordCloud (optional but cool)
wordcloud = WordCloud(font_path='kalpurush.ttf', width=800, height=400, background_color='white').generate(' '.join(all_words))

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Bangla Text")
plt.tight_layout()
plt.show()
