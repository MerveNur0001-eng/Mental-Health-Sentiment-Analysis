import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK resources
logging.info("Downloading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Create static folder if it doesn't exist
if not os.path.exists('static'):
    logging.info("Creating static folder...")
    os.makedirs('static')

# Load dataset
logging.info("Loading dataset...")
try:
    df = pd.read_csv("Combined_Data.csv", index_col="Unnamed: 0")
except FileNotFoundError:
    logging.error("Combined_Data.csv not found in the project directory.")
    print("Error: Combined_Data.csv not found. Please ensure the file is in C:\\Users\\Merve\\mental.")
    exit(1)
except Exception as e:
    logging.error(f"Failed to load dataset: {str(e)}")
    print(f"Error: Failed to load dataset: {str(e)}")
    exit(1)

# Data Cleanup
logging.info(f"Dataset loaded with {len(df)} rows.")
if not df.empty:
    logging.info("Checking for missing 'statement' and 'status' columns...")
    if 'statement' not in df.columns or 'status' not in df.columns:
        logging.error("Dataset missing required columns: 'statement' and/or 'status'.")
        print("Error: Dataset must contain 'statement' and 'status' columns.")
        exit(1)

    logging.info("Dropping rows with missing 'statement' values...")
    df = df.dropna(subset=['statement'])
    logging.info(f"Dataset after dropping missing values: {len(df)} rows.")

    # VADER Sentiment Analysis
    analyzer = SentimentIntensityAnalyzer()
    def get_sentiment(text):
        try:
            scores = analyzer.polarity_scores(str(text))
            compound = scores['compound']
            if compound >= 0.05:
                return 'Positive'
            elif compound <= -0.05:
                return 'Negative'
            else:
                return 'Neutral'
        except Exception as e:
            logging.error(f"Error in get_sentiment for text '{text}': {str(e)}")
            return 'Neutral'

    logging.info("Applying VADER sentiment analysis...")
    start_time = time.time()
    df['sentiment'] = df['statement'].apply(get_sentiment)
    df['score'] = df['statement'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
    logging.info(f"VADER sentiment analysis completed in {time.time() - start_time:.2f} seconds.")

    # Text Cleaning
    def clean_text(text):
        try:
            text = str(text).lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
            return ' '.join(tokens)
        except Exception as e:
            logging.error(f"Error in clean_text for text '{text}': {str(e)}")
            return ''

    logging.info("Cleaning text data...")
    start_time = time.time()
    df['cleaned_statement'] = df['statement'].apply(clean_text)
    logging.info(f"Text cleaning completed in {time.time() - start_time:.2f} seconds.")

    # Check for empty cleaned statements
    if df['cleaned_statement'].str.strip().eq('').any():
        logging.warning("Some cleaned statements are empty after processing.")

    # Features and Targets
    logging.info("Preparing features and targets...")
    X = df['cleaned_statement']
    y_status = df['status']
    y_sentiment = df['sentiment']

    # Train-test split
    logging.info("Splitting data...")
    X_status_train, X_status_test, y_status_train, y_status_test = train_test_split(
        X, y_status, test_size=0.25, random_state=1
    )
    X_sentiment_train, X_sentiment_test, y_sentiment_train, y_sentiment_test = train_test_split(
        X, y_sentiment, test_size=0.25, random_state=1
    )

    # Status Pipeline
    logging.info("Training status pipeline...")
    status_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=None)),
        ('classifier', LinearSVC(dual='auto'))
    ])
    status_pipeline.fit(X_status_train, y_status_train)

    # Sentiment Pipeline
    logging.info("Training sentiment pipeline...")
    sentiment_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=None)),
        ('classifier', LinearSVC(dual='auto'))
    ])
    sentiment_pipeline.fit(X_sentiment_train, y_sentiment_train)

    # Evaluate Models
    logging.info("Evaluating models...")
    print("Status Model Evaluation:")
    print(classification_report(y_status_test, status_pipeline.predict(X_status_test)))
    print("Sentiment Model Evaluation:")
    print(classification_report(y_sentiment_test, sentiment_pipeline.predict(X_sentiment_test)))

    # Save Models
    logging.info("Saving models...")
    with open('status_pipeline.pkl', 'wb') as f:
        pickle.dump(status_pipeline, f)
    with open('sentiment_pipeline.pkl', 'wb') as f:
        pickle.dump(sentiment_pipeline, f)

    # Visualize Sentiment Distribution
    logging.info("Generating sentiment distribution plot...")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='sentiment', data=df, palette=['#34D399', '#F87171', '#60A5FA'])
    plt.title('Sentiment Distribution in Dataset')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.savefig('static/sentiment_distribution.png')
    plt.close()
    logging.info("Script completed successfully.")
else:
    logging.error("No data available in dataset.")
    print("Error: No data available for training. Application will use VADER for predictions.")