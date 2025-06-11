### May 13, 2025
- Created GitHub repository for the project: `bangla-headline-mismatch`
- Cloned the repository locally and initialized folder structure

### May 14, 2025
- Colledcted 16 samples for dataset from various Bangladeshi news portals.

### May 15, 2025
- Colledcted 6 new samples for dataset from various Bangladeshi news portals.
- Wrote preprocessing script for Bangla text.
- Removed punctuation and cleaned Bangla headlines and bodies.
- Saved cleaned file as `data/cleaned_dataset.csv`.

### May 16, 2025
- Completed TF-IDF and Bag of Words feature extraction.
- Fixed file path issues and verified proper data loading.
- Faced package installation errors (e.g., gensim, scipy) — decided to switch to Anaconda for smoother setup.

### May 10, 2025
- Split dataset into training and testing sets using train_test_split with random_state=42.

- Trained a Logistic Regression model on TF-IDF features.

- Evaluated the model with a classification report (accuracy: 40% on a 5-sample test set).

- Saved trained model and vectorizer to models/model.pkl and models/vectorizer.pkl.

- Created an inference script predict.py that loads the model and predicts match/mismatch on new input.

- Verified working prediction with sample input.


