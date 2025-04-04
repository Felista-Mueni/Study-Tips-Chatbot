import streamlit as st
import nltk
from nltk import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Download necessary NLTK resources
nltk.download('punkt')

# Function to preprocess the text
def preprocess(text):
    """
    This function processes the raw text by:
    - Removing unnecessary newlines and extra spaces.
    - Tokenizing the text into sentences.
    - Cleaning each sentence by removing unnecessary characters, including `*`.
    """
    # Remove newline characters and extra spaces
    text = re.sub(r'\r\n', ' ', text)  # Replace newlines with space
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space

    # Remove the `*` symbols and any other non-alphanumeric characters except for punctuation
    text = re.sub(r'\*', '', text)  # Remove star symbols
    text = re.sub(r'[^a-zA-Z0-9\s.,?!]', '', text)  # Remove other unwanted characters

    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    return sentences

# Function to compute the most relevant sentence
def get_most_relevant_sentence(query, sentences):
    vectorizer = TfidfVectorizer().fit_transform([query] + sentences)
    cosine_similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:])
    most_similar_idx = cosine_similarities.argmax()
    return sentences[most_similar_idx]

# Chatbot function

def chatbot(query, file_path = 'Study Tips.txt'):
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        text = file.read()
    sentences = preprocess(text)
    response = get_most_relevant_sentence(query, sentences)
    return response

# Streamlit app definition
def main():
    st.title("Student Chatbot")
    text_file_path = "Study Tips.txt"  # Adjust path if necessary
    user_query = st.text_input("Ask me a question about study tips:")
    if user_query:
        response = chatbot(user_query, text_file_path)
        st.write("Chatbot Response:", response)

# Run the Streamlit app
if __name__ == "__main__":
    main()
