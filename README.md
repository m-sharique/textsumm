# Text Summarization Project (textsumm)

This project implements a text summarization application using Streamlit and TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.

## Overview

Text summarization is the process of distilling the most important information from a source text to produce a concise summary. This project utilizes TF-IDF to generate summaries based on the input text provided by the user.

## Features

- **Summarization:** Input any text and generate a summary using TF-IDF.
- **Interactive Web Interface:** Built with Streamlit for a user-friendly experience.

## Project Structure

The project is structured into two main parts:

1. **App.py:** Contains the Streamlit web application for text summarization.
2. **Model_prepare.py:** Prepares the TF-IDF vectorizer using the CNN/DailyMail dataset.

## Installation

To run the application locally, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/m-sharique/textsumm.git
   cd textsumm
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Download NLTK data:
   ```
   python -m nltk.downloader punkt
   python -m nltk.downloader stopwords
   ```

4. Run the Streamlit app:
   ```
   streamlit run App.py
   ```

## Usage

1. Enter or paste the text you want to summarize into the text area.
2. Click the "Summarize" button to generate a summary based on TF-IDF.

## Requirements

The project requires the following Python libraries (see `requirements.txt` for details):

- Streamlit
- nltk
- scikit-learn
- joblib
- datasets
- rouge_score

## Future Enhancements

- Evaluation functionality using ROUGE scores.
- Improved handling of different text formats and languages.
- Integration with more advanced summarization algorithms (e.g., BERT-based models).

## Contributors

- Mohammed Sharique Siddiqui(https://github.com/m-sharique)
