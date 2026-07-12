import random
import nltk
import pandas as pd
import joblib  # Production library for model saving
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

print("Loading dataset...")
df = pd.read_csv('my_dataset.csv')
text_col = 'text' if 'text' in df.columns else 'review' if 'review' in df.columns else None
label_col = 'sentiment' if 'sentiment' in df.columns else 'label' if 'label' in df.columns else None
df = df.dropna(subset=[text_col, label_col])

print("Preprocessing text...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

negation_words = {'not', 'no', 'never', 'neither', 'nor', 'but', 'without', 'against'}
stop_words = set(stopwords.words('english')) - negation_words
lemmatizer = WordNetLemmatizer()

def preprocess_text(words):
    transformed_words, negate_counter = [], 0
    reset_words = {'.', ',', ';', '!', '?', 'but', 'and', 'however', 'although', 'yet', 'br', 'html'}
    for word in words:
        word_lower = word.lower()
        if word_lower in negation_words:
            negate_counter = 3
            transformed_words.append(word_lower)
        elif word_lower in reset_words or len(word_lower) > 15:
            negate_counter = 0
            transformed_words.append(word_lower)
        elif negate_counter > 0:
            transformed_words.append(f"not_{word_lower}")
            negate_counter -= 1
        else:
            transformed_words.append(word_lower)
    final_words = []
    for word in transformed_words:
        if word.startswith("not_"):
            base = word[4:]
            if base.isalpha(): final_words.append(f"not_{lemmatizer.lemmatize(base)}")
        else:
            if word.isalpha() and word not in stop_words: final_words.append(lemmatizer.lemmatize(word))
    return ' '.join(final_words)

label_mapping = {'positive': 'pos', 'pos': 'pos', '1': 'pos', 1: 'pos', 'negative': 'neg', 'neg': 'neg', '0': 'neg', 0: 'neg'}
processed_reviews = []
for raw_text, raw_label in zip(df[text_col], df[label_col]):
    clean_label = label_mapping.get(str(raw_label).strip().lower(), str(raw_label).strip().lower())
    processed_reviews.append((preprocess_text(word_tokenize(str(raw_text))), clean_label))

random.seed(42)
random.shuffle(processed_reviews)

print("Extracting features...")
X = [r[0] for r in processed_reviews]
y = [r[1] for r in processed_reviews]
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)
X = vectorizer.fit_transform(X)

print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = LogisticRegression(C=5.0, class_weight='balanced', max_iter=1000)
classifier.fit(X_train, y_train)

# ==========================================
# EXPORT STEP: Save artifacts to disk
# ==========================================
print("\nSerializing and saving production files...")
joblib.dump(classifier, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("-> Successfully saved 'sentiment_model.pkl' & 'tfidf_vectorizer.pkl'!")