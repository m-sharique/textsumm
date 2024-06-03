import numpy as np
import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if not w.lower() in stop_words]
    return ' '.join(filtered_text)

# Function to sample 50% of the dataset
def sample_data(dataset, sample_size=0.5):
    total_size = len(dataset)
    sample_size = int(total_size * sample_size)
    sampled_indices = random.sample(range(total_size), sample_size)
    return dataset.select(sampled_indices)

# Load and sample the dataset
def load_and_sample_dataset():
    dataset = load_dataset('cnn_dailymail', '3.0.0')
    train_data = dataset['train']
    train_data = sample_data(train_data, 0.5)
    return train_data

# Preprocess the dataset
def preprocess_dataset(train_data):
    train_data = train_data.map(lambda x: {'article': preprocess(x['article']), 'highlights': preprocess(x['highlights'])})
    return train_data

# Load, sample, and preprocess the dataset
train_data = load_and_sample_dataset()
train_data = preprocess_dataset(train_data)

# Prepare training data
articles = train_data['article']
summaries = train_data['highlights']

# Train the model
def train_model(articles):
    vectorizer = TfidfVectorizer(max_df=0.85, max_features=10000)
    X = vectorizer.fit_transform(articles)
    return vectorizer

vectorizer = train_model(articles)

# Save the trained model
joblib.dump(vectorizer, 'model.joblib')

# Save the training data for later use in evaluation
joblib.dump((articles, summaries), 'training_data.joblib')
