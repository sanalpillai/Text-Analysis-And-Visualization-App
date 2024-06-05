import streamlit as st
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

# Function to preprocess text
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return filtered_tokens

# Function to generate word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stop_words, 
                min_font_size = 10).generate(text)
    return wordcloud

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
            plt.figure(figsize = (8, 8), facecolor = None) 
            plt.imshow(wordcloud) 
            plt.axis("off") 
            st.pyplot(plt)

# Run the app
if __name__ == '__main__':
    main()
