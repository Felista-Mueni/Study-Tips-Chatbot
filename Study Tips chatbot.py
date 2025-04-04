import streamlit as st
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Download necessary NLTK resources
nltk.download('punkt')

# Function to preprocess the text
def preprocess(text):
    text = re.sub(r'\r\n', ' ', text)  # Replace newline characters with space
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    sentences = nltk.sent_tokenize(text)
    sentences = [re.sub(r'[^a-zA-Z0-9\s]', '', sentence) for sentence in sentences]
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
