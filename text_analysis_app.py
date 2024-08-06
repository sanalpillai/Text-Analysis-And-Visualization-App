import os
import subprocess
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

# Set the path to ffmpeg
os.environ['FFMPEG_BINARY'] = '/usr/bin/ffmpeg'  # Update this path to where ffmpeg is located

# Set page config
st.set_page_config(page_title="Text & Video Analysis App", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for dark mode
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100;300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
        color: #e0e0e0;
    }
    
    .main {
        background-color: #1e1e1e;
    }
    
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    h1, h2, h3 {
        color: #bb86fc;
    }
    
    h1 {
        font-weight: 700;
        padding-bottom: 20px;
        border-bottom: 2px solid #bb86fc;
        margin-bottom: 30px;
    }
    
    .stButton>button {
        color: #ffffff;
        background-color: #bb86fc;
        border: none;
        border-radius: 4px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #a370f7;
    }
    
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #2e2e2e;
        color: #e0e0e0;
        border: 1px solid #4e4e4e;
        border-radius: 4px;
        padding: 10px;
    }
    
    .css-1v0mbdj.ebxwdo61 {
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        border-radius: 8px;
        padding: 20px;
        background-color: #2e2e2e;
    }
    
    .stSlider>div>div>div>div {
        background-color: #bb86fc;
    }
    
    .stProgress>div>div>div>div {
        background-color: #bb86fc;
    }
    
    .stRadio>div {
        background-color: #2e2e2e;
        border-radius: 4px;
        padding: 10px;
    }
    
    .st-emotion-cache-10trblm {
        color: #e0e0e0;
    }
    
    .st-emotion-cache-1gulkj5 {
        background-color: #2e2e2e;
    }
    
    .word-freq-header {
        color: #bb86fc;
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

load_nltk_data()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.isalnum() and word not in stop_words]

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='#1e1e1e', 
                          stopwords=set(stopwords.words('english')), 
                          min_font_size=10, colormap='viridis').generate(text)
    return wordcloud

def analyze_sentiment(text):
    return TextBlob(text).sentiment

def summarize_lsa(text, summarization_ratio=0.2):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer(Stemmer("english"))
    summarizer.stop_words = get_stop_words("english")
    sentence_count = max(1, int(summarization_ratio * len(parser.document.sentences)))
    return " ".join(str(sentence) for sentence in summarizer(parser.document, sentences_count=sentence_count))

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
            'ffmpeg_location': '/usr/bin/ffmpeg',  # Specify the ffmpeg binary path
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_id = info_dict.get('id', 'video')
            audio_file = f'downloads/{video_id}.wav'

        st.write(f"Attempting to access audio file: {audio_file}")
        
        if not os.path.exists(audio_file):
            st.error(f"Audio file not found at {audio_file}. Checking directory contents:")
            directory = 'downloads'
            files = os.listdir(directory)
            for file in files:
                st.write(f"- {file}")
            return None

        st.success(f"Audio file found: {audio_file}")
        
        model = whisper.load_model("base")
        result = model.transcribe(audio_file)
        return result["text"]
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None
    
def analyze_text(text, summarization_ratio):
    if not text:
        st.error("Please enter text or transcribe a video before analyzing.")
        return

    st.subheader("Analysis Results")

    col1, col2 = st.columns([3, 2])

    with col1:
        with st.expander("Analyzed Text", expanded=True):
            st.text_area("", text, height=200)

        tokens = preprocess_text(text)
        word_freq = Counter(tokens).most_common(10)
        
        st.markdown('<p class="word-freq-header">Top 10 Word Frequencies</p>', unsafe_allow_html=True)
        with st.expander("View Frequencies", expanded=True):
            for word, freq in word_freq:
                st.text(f"{word}: {freq}")

    with col2:
        st.subheader('Word Cloud')
        wordcloud = generate_wordcloud(' '.join(tokens))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        fig.patch.set_facecolor('#1e1e1e')
        st.pyplot(fig)

    st.subheader('Sentiment Analysis')
    sentiment = analyze_sentiment(text)
    col1, col2 = st.columns(2)
    col1.metric("Polarity", f"{sentiment.polarity:.2f}")
    col2.metric("Subjectivity", f"{sentiment.subjectivity:.2f}")

    # Generate one-line summary for sentiment analysis
    polarity = sentiment.polarity
    subjectivity = sentiment.subjectivity
    
    polarity_label = "neutral"
    if polarity > 0.1:
        polarity_label = "positive"
    elif polarity < -0.1:
        polarity_label = "negative"
    
    subjectivity_label = "neutral"
    if subjectivity > 0.6:
        subjectivity_label = "highly subjective"
    elif subjectivity < 0.4:
        subjectivity_label = "quite objective"

    sentiment_summary = f"The text is {polarity_label} in tone and {subjectivity_label} in nature."
    st.markdown(f"**Sentiment Summary:** {sentiment_summary}")

    st.subheader('Text Summary')
    summary = summarize_lsa(text, summarization_ratio)
    st.text_area("", summary, height=150)

def check_ffmpeg():
    try:
        result = subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True, text=True)
        st.write("FFmpeg version: " + result.stdout.split('\n')[0])
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"FFmpeg check failed: {str(e)}")
        return False
    except FileNotFoundError:
        st.error("FFmpeg not found in system path")
        return False
    
def main():
    st.title('Text & Video Analysis App')

    input_method = st.radio('Select input method:', ('Text', 'YouTube Video'), horizontal=True)

    if input_method == 'Text':
        text = st.text_area('Enter your text here:', height=200)
    else:
        video_url = st.text_input('Enter YouTube video URL:')
        if 'transcribed_text' in st.session_state:
            text = st.text_area('Transcribed text:', st.session_state.transcribed_text, height=200)
        else:
            text = ""
        
        if st.button('Transcribe Video'):
            if video_url:
                with st.spinner('Transcribing video...'):
                    transcribed_text = transcribe_youtube_video(video_url)
                if transcribed_text:
                    st.session_state.transcribed_text = transcribed_text
                    st.success("Transcription successful! The transcribed text is displayed below. Click 'Analyze' to proceed.")
                    st.text_area('Transcribed text:', transcribed_text, height=200)
                    text = transcribed_text
            else:
                st.error("Please enter a valid YouTube URL.")

    summarization_ratio = st.slider('Summarization ratio', 0.1, 1.0, 0.2)

    if st.button('Analyze'):
        analyze_text(text, summarization_ratio)

if __name__ == '__main__':
    main()
