from flask import Flask, request, render_template
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK resources
logging.info("Downloading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize VADER for fallback predictions
analyzer = SentimentIntensityAnalyzer()

# Load models (if available)
try:
    with open('status_pipeline.pkl', 'rb') as f:
        status_pipeline = pickle.load(f)
    with open('sentiment_pipeline.pkl', 'rb') as f:
        sentiment_pipeline = pickle.load(f)
    logging.info("Models loaded successfully.")
except FileNotFoundError:
    logging.warning("Model files not found. Using VADER for predictions.")
    status_pipeline = None
    sentiment_pipeline = None

# Text preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# VADER-based prediction (fallback)
def vader_predict(text):
    scores = analyzer.polarity_scores(str(text))
    compound = scores['compound']
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Rule-based suggestions
def get_rule_based_suggestions(sentiment, status=None):
    if status is None:  # Fallback mode
        if sentiment == 'Positive':
            return "You're doing great! Keep up positive habits like journaling or spending time with loved ones. Consider writing down three things you're grateful for today."
        elif sentiment == 'Negative':
            return "It seems you're feeling down. Try deep breathing exercises: inhale for 4 seconds, hold for 4, exhale for 4. You can also reach out to a mental health professional. Visit <a href='https://www.mentalhealth.gov' target='_blank' class='text-blue-600 underline'>Mental Health Resources</a> for support."
        else:
            return "Your mood seems balanced. Consider mindfulness practices, like a 5-minute meditation, to stay grounded."
    else:
        if sentiment == 'Positive' and status == 'Normal':
            return "Great to hear you're feeling positive! Try daily gratitude journaling or a short walk to maintain your well-being."
        elif sentiment == 'Negative' or status in ['Depression', 'Anxiety']:
            return "You might be struggling. Consider mindfulness exercises, talking to a trusted friend, or seeking professional help. Visit <a href='https://www.mentalhealth.gov' target='_blank' class='text-blue-600 underline'>Mental Health Resources</a> for support."
        elif status in ['Normal'] and sentiment == 'Neutral':
            return "Your mood seems stable. Regular exercise or a brief meditation session can help maintain this balance."
        else:
            return "Your input suggests a mixed state. Try relaxation techniques like deep breathing or consult a professional for personalized advice. Visit <a href='https://www.mentalhealth.gov' target='_blank' class='text-blue-600 underline'>Mental Health Resources</a>."

@app.route('/')
def home():
    plot_exists = os.path.exists('static/sentiment_distribution.png')
    logging.info(f"Plot exists: {plot_exists}")
    return render_template('index.html', plot_exists=plot_exists)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        statement = request.form['statement']
        if not statement.strip():
            logging.warning("Empty statement submitted.")
            return render_template('error.html', error_message="Please enter a valid statement.")
        
        cleaned_statement = clean_text(statement)
        logging.info(f"Processing statement: {statement}")

        if status_pipeline is None or sentiment_pipeline is None:
            # Fallback to VADER
            sentiment = vader_predict(statement)
            status = None
            suggestions = get_rule_based_suggestions(sentiment)
            logging.info(f"VADER prediction - Sentiment: {sentiment}")
        else:
            # Model-based predictions
            sentiment = sentiment_pipeline.predict([cleaned_statement])[0]
            status = status_pipeline.predict([cleaned_statement])[0]
            suggestions = get_rule_based_suggestions(sentiment, status)
            logging.info(f"Model prediction - Sentiment: {sentiment}, Status: {status}")

        return render_template('index.html', 
                             statement=statement, 
                             sentiment=sentiment, 
                             status=status if status else "Not available (model not loaded)", 
                             suggestions=suggestions)
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return render_template('error.html', error_message=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)