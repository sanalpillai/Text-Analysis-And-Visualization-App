import streamlit as st
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

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

# Function to summarize text using LSA
def summarize_lsa(text, summarization_ratio=0.2):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    stemmer = Stemmer("english")
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words("english")
    summary = summarizer(parser.document, sentences_count=int(summarization_ratio * len(text.split('\n'))))
    return " ".join([str(sentence) for sentence in summary])

# Main Streamlit app
def main():
    st.title('Text Analysis, Visualization, and Study Planner')

    # Input text area
    text = st.text_area('Enter your text here:')

    # Select summarization ratio
    summarization_ratio = st.slider('Select summarization ratio', 0.1, 1.0, 0.2)

    # Cache the result of the summarization
    @st.cache_data(show_spinner=False)
    def cached_summarization(text, summarization_ratio):
        summary = summarize_lsa(text, summarization_ratio=summarization_ratio)
        return summary

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

            # Summarization
            st.subheader('Text Summarization')
            summary = cached_summarization(text, summarization_ratio)
            st.write(summary)

            # Visualization Options
            st.subheader('Visualization Options')
            st.write('### Word Frequency Bar Chart')
            word_freq_df = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency'])
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Frequency', y='Word', data=word_freq_df.sort_values(by='Frequency', ascending=False).head(20))
            plt.title('Top 20 Words by Frequency')
            st.pyplot(plt)

            st.write('### Word Frequency Table')
            st.table(word_freq_df.sort_values(by='Frequency', ascending=False).head(20))

# Run the app
if __name__ == '__main__':
    main()
