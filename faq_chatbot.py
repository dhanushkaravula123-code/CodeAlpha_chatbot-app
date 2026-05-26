import pandas as pd
import nltk
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# download nltk data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# load dataset
df = pd.read_csv("faqs.csv")

# preprocessing function
def preprocess_text(text):

    text = text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))

    words = word_tokenize(text)

    stop_words = set(stopwords.words('english'))

    filtered_words = [word for word in words if word not in stop_words]

    return " ".join(filtered_words)

# preprocess all FAQ questions
df['processed_question'] = df['question'].apply(preprocess_text)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(df['processed_question'])

# chatbot function
def chatbot(user_question):

    processed_question = preprocess_text(user_question)

    user_vector = vectorizer.transform([processed_question])

    similarity = cosine_similarity(user_vector, faq_vectors)

    best_match_index = similarity.argmax()

    best_score = similarity[0][best_match_index]

    if best_score > 0.3:
        return df.iloc[best_match_index]['answer']
    else:
        return "Sorry, I don't understand the question."

# chat loop
print("FAQ Chatbot Started!")
print("Type 'exit' to stop.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == 'exit':
        print("Chatbot ended.")
        break

    response = chatbot(user_input)

    print("Bot:", response)