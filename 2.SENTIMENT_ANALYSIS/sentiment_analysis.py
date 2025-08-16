import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle

# -------------------------
# Load and preprocess data
# -------------------------
@st.cache_data(show_spinner=True)
def load_and_prepare_data(filepath, sample_size=20000):
    """
    Loads the CSV file and performs initial preprocessing.
    - Selects only necessary columns.
    - Filters out neutral sentiments.
    - Maps 0 to 'negative' and 4 to 'positive'.
    - Shuffles data and takes a sample if needed (for performance).
    """
    df = pd.read_csv(filepath, encoding='latin-1', names=['sentiment', 'id', 'date', 'query', 'user', 'text'])

    # Filter only positive and negative samples
    df = df[df['sentiment'].isin([0, 4])]

    # Map numeric labels to strings for better readability
    df['sentiment'] = df['sentiment'].map({0: 'negative', 4: 'positive'})
    df.head(10)
    # Drop unnecessary columns and retain only text + sentiment
    df = df[['text', 'sentiment']]

    # Shuffle the dataset to avoid training bias
    df = shuffle(df, random_state=42).reset_index(drop=True)

    # If too large, sample for speed (adjust sample_size if needed)
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    return df

# -------------------------
# Train machine learning model
# -------------------------
@st.cache_resource(show_spinner=True)
def train_model(df):
    """
    Trains a sentiment classification model using:
    - TF-IDF Vectorization
    - Logistic Regression Classifier
    - Improves generalization by adjusting hyperparameters
    - Returns pipeline, accuracy, and classification metrics
    """
    X = df['text']            # Feature = text data
    y = df['sentiment']       # Target = sentiment label

    # Split the dataset into train and test (80-20 ratio)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build pipeline: TF-IDF vectorizer + Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',         # Remove common stopwords
            max_features=15000,           # Increase features for better text representation
            ngram_range=(1, 2)            # Use both unigrams and bigrams
        )),
        ('clf', LogisticRegression(
            C=1.5,                        # Regularization strength
            solver='lbfgs',               # Solver for optimization
            max_iter=500,                 # More iterations to ensure convergence
            class_weight='balanced'       # Handle imbalance in class labels
        ))
    ])

    # Fit the model on training data
    pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)

    # Evaluate model performance
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return pipeline, acc, report

# -------------------------
# Streamlit chatbot interface
# -------------------------
def main():
    # Basic Streamlit page config
    st.set_page_config(page_title="Sentiment Analysis Chatbot", page_icon="💬", layout="wide")

    # Inject basic CSS for dashboard appearance and chat bubbles
    st.markdown("""
        <style>
        .main-container {
            background-color: #f7f9fb;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0px 0px 15px rgba(0,0,0,0.1);
        }
        .user-msg {
            background-color: #0f62fe;
            color: white;
            padding: 12px 20px;
            border-radius: 20px;
            margin: 10px 0;
            text-align: right;
            max-width: 75%;
            float: right;
            clear: both;
        }
        .bot-msg {
            background-color: #333;
            color: white;
            padding: 12px 20px;
            border-radius: 20px;
            margin: 10px 0;
            text-align: left;
            max-width: 75%;
            float: left;
            clear: both;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and divider
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #0f62fe;'>Sentiment Analysis Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # Load dataset and train model
    with st.spinner('Loading and training the model...'):
        df = load_and_prepare_data('sentiment_data.csv', sample_size=None)  # use full dataset
        model, accuracy, class_report = train_model(df)

    # Display model performance
    st.success(f"Model accuracy: {accuracy*100:.2f}%")

    with st.expander("Click to view detailed classification report"):
        st.json(class_report)

    # Initialize chat history if not already done
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Function to generate response based on sentiment
    def get_bot_reply(user_text):
        pred = model.predict([user_text])[0]
        reply_dict = {
            'positive': "Glad to hear that! 😊",
            'negative': "Sorry to hear that. Hope things get better. 💙",
        }
        return reply_dict.get(pred, "Thanks for sharing."), pred

    # Input text box for user message
    user_input = st.text_input("Type your message:", "")

    # Process user input
    if user_input:
        st.session_state.chat_history.append({"sender": "user", "text": user_input})
        bot_reply, sentiment = get_bot_reply(user_input)
        st.session_state.chat_history.append({"sender": "bot", "text": bot_reply, "sentiment": sentiment})

    # Display chat bubbles from history
    for chat in st.session_state.chat_history:
        if chat['sender'] == 'user':
            st.markdown(f"<div class='user-msg'>{chat['text']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'>{chat['text']}<br><small style='color:#0f62fe;'>Sentiment: {chat['sentiment'].capitalize()}</small></div>", unsafe_allow_html=True)

    # Close main container
    st.markdown("</div>", unsafe_allow_html=True)

# Entry point of the application
if __name__ == "__main__":
    main()