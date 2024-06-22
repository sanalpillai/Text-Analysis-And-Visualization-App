import os
import streamlit as st
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import yt_dlp
import whisper
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

# Function to summarize text using LSA
def summarize_lsa(text, summarization_ratio=0.2):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    stemmer = Stemmer("english")
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words("english")
    summary = summarizer(parser.document, sentences_count=int(summarization_ratio * len(text.split('\n'))))
    return " ".join([str(sentence) for sentence in summary])

# Function to transcribe YouTube video
@st.cache_data(show_spinner=False)
def transcribe_youtube_video(url):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'downloads/%(id)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            audio_file = ydl.prepare_filename(info_dict).replace('.webm', '.wav')

        if not os.path.exists(audio_file):
            st.error("Audio file was not created. Please check the ffmpeg installation.")
            return None

        model = whisper.load_model("base")
        result = model.transcribe(audio_file)
        return result["text"]
    except yt_dlp.utils.DownloadError as e:
        st.error(f"Error downloading video: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Main Streamlit app
def main():
    st.title('Text & Video Analysis App')

    # Input method selection
    input_method = st.radio('Select input method:', ('Text', 'YouTube Video'))

    # Text input
    if input_method == 'Text':
        text = st.text_area('Enter your text here:')
    else:
        video_url = st.text_input('Enter YouTube video URL:')
        if st.button('Transcribe Video'):
            if video_url:
                transcribed_text = transcribe_youtube_video(video_url)
                if transcribed_text:
                    st.session_state.transcribed_text = transcribed_text
                    st.write("Transcription successful! Click the 'Analyze' button to proceed.")
            else:
                st.error("Please enter a valid YouTube URL.")

    if input_method == 'YouTube Video' and 'transcribed_text' in st.session_state:
        text = st.session_state.transcribed_text

        st.write("Transcribed Text:")
        st.text_area("Transcription", text, key="transcription_text")

    # Select summarization ratio
    summarization_ratio = st.slider('Select summarization ratio', 0.1, 1.0, 0.2)

    # Analyze text
    def analyze_text(text, summarization_ratio):
        if not text and 'transcribed_text' in st.session_state:
            text = st.session_state.transcribed_text

        if text:
            # Display the transcribed text
            st.write("Transcribed Text:")
            st.text_area("Transcription", text)

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
            st.write(f"Sentiment: {sentiment}")
            
            # Text summarization
            summary_text = text if 'Transcription' not in text else st.session_state.transcribed_text
            summary = summarize_lsa(summary_text, summarization_ratio)
            st.write("Summary:")
            st.text_area("Summary", summary)

    if st.button('Analyze'):
        analyze_text(text, summarization_ratio)
