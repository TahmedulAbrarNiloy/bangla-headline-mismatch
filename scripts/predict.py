import pickle

# Load model
with open('saved_model/logistic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load vectorizer
with open('saved_model/tfidf_vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Example inputs (replace these with your own)
headline = "ঐকমত্য কমিশনের আজকের বৈঠকে অংশ নেয়নি জামায়াত"
body ="জাতীয় ঐকমত্য কমিশনের সঙ্গে রাজনৈতিক দলগুলোর দ্বিতীয় পর্যায়ের অসমাপ্ত আলোচনায় আজ মঙ্গলবার অংশ নেয়নি বাংলাদেশ জামায়াতে ইসলামী। তবে বিএনপিসহ অন্য রাজনৈতিক দলগুলো আলোচনায় অংশ নিচ্ছে।"

# Preprocess (if needed)
# You can import and use your existing clean_text function here

# Combine inputs as you did during training
combined = headline + " " + body
vectorized_input = vectorizer.transform([combined])

# Predict
prediction = model.predict(vectorized_input)[0]

print("Prediction:", "Matched" if prediction == 1 else "Mismatched")