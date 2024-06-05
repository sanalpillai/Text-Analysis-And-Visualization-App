import streamlit as st
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
from textblob import TextBlob

nltk.download('stopwords')
nltk.download('punkt')

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return filtered_tokens

# Function to generate word cloud
def generate_wordcloud(text):
    stop_words = set(stopwords.words('english'))
    wordcloud = WordCloud(width=800, height=800, background_color='white', stopwords=stop_words, min_font_size=10).generate(text)
    return wordcloud

# Function for sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment

# Main Streamlit app
def main():
    st.title('Text Analysis and Visualization')

    # Input text area
    text = st.text_area('Enter your text here:')

    if st.button('Analyze'):
        if text:
            # Preprocess the text
            tokens = preprocess_text(text)
            # Word frequency analysis
            word_freq = Counter(tokens)
            st.subheader('Word Frequency Analysis')
            st.write(word_freq)

            # Word cloud
            st.subheader('Word Cloud')
            wordcloud = generate_wordcloud(' '.join(tokens))
            plt.figure(figsize=(8, 8), facecolor=None)
            plt.imshow(wordcloud)
            plt.axis("off")
            st.pyplot(plt)

            # Sentiment analysis
            st.subheader('Sentiment Analysis')
            sentiment = analyze_sentiment(text)

            # Display sentiment analysis results in an easier to understand format
            sentiment_str = ""
            if sentiment.polarity > 0:
                sentiment_str += "Positive"
            elif sentiment.polarity < 0:
                sentiment_str += "Negative"
            else:
                sentiment_str += "Neutral"

            st.write(f"Sentiment: {sentiment_str}")
            st.write(f"Polarity: {sentiment.polarity:.2f} (from -1 to 1, where 1 means positive sentiment)")
            st.write(f"Subjectivity: {sentiment.subjectivity:.2f} (from 0 to 1, where 0 means objective and 1 means subjective)")

# Run the app
if __name__ == '__main__':
    main()
