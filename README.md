# Movie Review Sentiment Analysis

## Project Description
This project is a **Sentiment Analysis system** that classifies movie reviews as **Positive** or **Negative**.  
It demonstrates how **Natural Language Processing (NLP)** and **Machine Learning** can be used to analyze textual data and understand human opinions.  

The system is built using **Python**, **NLTK**, and **Scikit-learn**. It preprocesses text by removing noise such as HTML tags, punctuation, and stopwords, converts the text into numerical features using CountVectorizer, and trains a **Naive Bayes classifier** to predict sentiment.  

This project is beginner-friendly and suitable for learning the fundamentals of NLP, text preprocessing, and machine learning.

---

## Key Tasks
1. Load the **IMDB movie review dataset** (CSV file).  
2. **Clean and preprocess** text data: lowercase, remove HTML tags, punctuation, and stopwords.  
3. **Split the dataset** into training (80%) and testing (20%) sets.  
4. Convert text into **numerical features** using CountVectorizer.  
5. **Train a Naive Bayes model** on the training data.  
6. Predict sentiment (Positive / Negative) for new reviews in **interactive mode**.  

---

## Sample Output

Enter your review (type 'exit' to quit):
Review: I loved the movie, it was amazing!
Prediction: Positive

Review: The movie was boring and too long.
Prediction: Negative

Review: exit
