import pickle

# Load model
with open('saved_model/logistic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load vectorizer
with open('saved_model/tfidf_vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Example inputs (replace these with your own)
headline = "শিক্ষা মন্ত্রণালয়ের নতুন সিদ্ধান্ত"
body = "সূর্যের এমন লকডাউনের ফলে বিশ্বে তাপমাত্রা কমে যাবে, শীতল হয়ে উঠবে পৃথিবী। এছাড়া বিশ্বজুড়ে ভূমিকম্প ও দুর্ভিক্ষের মতো ভয়ংকর দুর্যোগ দেখা দিতে পারে বলে শঙ্কা প্রকাশ করেছেন বিজ্ঞানীরা।নিউইয়র্ক টাইমসের প্রতিবেদনে বলা হয়েছে, সূর্য বর্তমানে ‘সোলার মিনিমাম’ পরিস্থিতিতে রয়েছে। ফলে পৃথিবীতে সূর্যের স্বাভাবিক সময়ে সরবরাহ করা তাপমাত্রা অনেক কমে গেছে। পৃথিবীর প্রতি সূর্যের কার্যকলাপ নাটকীয়ভাবে হ্রাস পেয়েছে।বিশ্বখ্যাত জ্যোতির্বিজ্ঞানী ড. টনি ফিলিপস বলেন, আমরা এমন গভীরতম সময়ের ভেতরে প্রবেশ করতে যাচ্ছি যে সময়ে সূর্যের আলো কার্যত অদৃশ্য হয়ে যাবে। সূর্যের সোলার মিনিমাম চলছে। এটি অত্যন্ত গভীর। সানস্পট গণনা থেকে বোঝা যাচ্ছে এটি বিগত শতাব্দীর সবচেয়ে গভীরতম অবস্থানে রয়েছে। সূর্যের চৌম্বকীয় শক্তি দুর্বল হয়ে পড়েছে। এর মানে হলো সৌরজগতে অতিরিক্ত মহাজাগতিক শক্তির প্রবেশের আভাস"

# Preprocess (if needed)
# You can import and use your existing clean_text function here

# Combine inputs as you did during training
combined = headline + " " + body
vectorized_input = vectorizer.transform([combined])

# Predict
prediction = model.predict(vectorized_input)[0]

print("Prediction:", "Matched" if prediction == 1 else "Mismatched")