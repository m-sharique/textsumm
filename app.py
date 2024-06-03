import streamlit as st
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
# from rouge_score import rouge_scorer
import joblib

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the trained model and training data
vectorizer = joblib.load('model.joblib')
articles, summaries = joblib.load('training_data.joblib')

# Function to preprocess text
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if not w.lower() in stop_words]
    return ' '.join(filtered_text)

# Summarization function
def summarize(text, vectorizer):
    sentences = sent_tokenize(text)
    preprocessed_sentences = [preprocess(sentence) for sentence in sentences]
    tfidf_matrix = vectorizer.transform(preprocessed_sentences)
    sentence_scores = tfidf_matrix.sum(axis=1).A1
    top_sentence_indices = sentence_scores.argsort()[-3:][::-1]
    summary = ' '.join([sentences[i] for i in top_sentence_indices])
    return summary

# Function to evaluate the summarization model using ROUGE scores
# def evaluate_model(articles, summaries, vectorizer):
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#     scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
#     for article, reference_summary in zip(articles, summaries):
#         generated_summary = summarize(article, vectorizer)
#         score = scorer.score(reference_summary, generated_summary)
#         scores['rouge1'].append(score['rouge1'].fmeasure)
#         scores['rouge2'].append(score['rouge2'].fmeasure)
#         scores['rougeL'].append(score['rougeL'].fmeasure)
    
#     average_scores = {key: np.mean(value) for key, value in scores.items()}
#     return average_scores

# Streamlit app interface
st.title("Text Summarization App")

text = st.text_area("Enter text to summarize:")

if st.button("Summarize"):
    summary = summarize(text, vectorizer)
    st.write("Summary:")
    st.write(summary)

# if st.button("Evaluate Model"):
#     average_scores = evaluate_model(articles, summaries, vectorizer)
#     st.write("Model Evaluation Scores:")
#     st.write(f"ROUGE-1 Score: {average_scores['rouge1']:.4f}")
#     st.write(f"ROUGE-2 Score: {average_scores['rouge2']:.4f}")
#     st.write(f"ROUGE-L Score: {average_scores['rougeL']:.4f}")
