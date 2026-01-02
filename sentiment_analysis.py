# sentiment_analysis.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
df = pd.read_csv("IMDB_Dataset.csv")  # Make sure CSV file is in same folder
print("Dataset Loaded!")

# Preprocess text
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"<.*?>", "", text)  # remove HTML tags
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation & numbers
    words = text.split()
    words = [w for w in words if w not in stop_words]  # remove stopwords
    return " ".join(words)

df['clean_review'] = df['review'].apply(clean_text)

# Split data
X = df['clean_review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numbers
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Interactive Prediction
print("\nEnter your review (type 'exit' to quit):")
while True:
    review = input("Review: ")
    if review.lower() == "exit":
        break
    review_clean = clean_text(review)
    review_vec = vectorizer.transform([review_clean])
    prediction = model.predict(review_vec)[0]
    print("Prediction:", prediction)
