# IMPORTS

import pdfplumber
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK SETUP (Run once if needed)
# Uncomment these only the first time you run
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# 1️⃣ Resume Text Extraction

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text

# 2️⃣ Text Preprocessing

def preprocess_text(text):
    # Normalize unicode
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    cleaned_tokens = []
    for word in tokens:
        if word.isalpha() and word not in stop_words:
            word = lemmatizer.lemmatize(word)
            cleaned_tokens.append(word)

    return " ".join(cleaned_tokens)

# 3️⃣ TF-IDF + Cosine Similarity

def calculate_similarity(resume_text, jd_text):

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )

    # Fit on resume
    resume_vector = vectorizer.fit_transform([resume_text])
    jd_vector = vectorizer.transform([jd_text])

    similarity = cosine_similarity(resume_vector, jd_vector)

    score = similarity[0][0] * 100

    return round(score, 2)

