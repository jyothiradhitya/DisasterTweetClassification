import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load datasets
train_data = pd.read_csv("train1.csv")
test_data = pd.read_csv("test1.csv")

# Preprocessing
df_train = train_data.copy()
df_train = df_train.drop_duplicates(["text", "target"])
df_train = df_train.drop(index=df_train[df_train.duplicated('text', keep=False)].index)

# Fill missing values
missing_cols = ['keyword', 'location']
df_train[missing_cols] = df_train[missing_cols].fillna("None")

# Text preprocessing
import re
from nltk.corpus import stopwords
from textblob import Word
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

sw = stopwords.words('english')
df_train['text'] = df_train['text'].apply(lambda x: re.sub('(http|ftp|https)://\\S+|www\\.\\S+', '', x))  # Remove URLs
df_train['text'] = df_train['text'].apply(lambda x: re.sub('RT', '', x))  # Remove RT
df_train['text'] = df_train['text'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))  # Remove special characters
df_train['text'] = df_train['text'].apply(lambda x: " ".join(x.lower() for x in x.split() if x not in sw))  # Remove stopwords
df_train['text'] = df_train['text'].apply(lambda x: " ".join(Word(word).lemmatize() for word in x.split()))  # Lemmatize

# Split into train/test sets
X = df_train['text']
y = df_train['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vectorization using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train_tfidf, y_train)

# Predictions
y_pred = log_reg.predict(X_test_tfidf)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model and vectorizer for future use
import joblib
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(log_reg, 'logistic_regression_model.pkl')

# Predict on new test data
test_data['text'] = test_data['text'].apply(lambda x: re.sub('(http|ftp|https)://\\S+|www\\.\\S+', '', x))
test_data['text'] = test_data['text'].apply(lambda x: re.sub('RT', '', x))
test_data['text'] = test_data['text'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
test_data['text'] = test_data['text'].apply(lambda x: " ".join(x.lower() for x in x.split() if x not in sw))
test_data['text'] = test_data['text'].apply(lambda x: " ".join(Word(word).lemmatize() for word in x.split()))

X_test_data_tfidf = tfidf_vectorizer.transform(test_data['text'])
test_data['predicted_target'] = log_reg.predict(X_test_data_tfidf)

# Save predictions
test_data[['id', 'predicted_target']].to_csv("submission.csv", index=False)
