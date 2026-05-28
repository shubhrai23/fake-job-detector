import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

print("Loading dataset...")
# Load the Kaggle dataset
df = pd.read_csv('fake_job_postings.csv')

# Phase 1: Data Preprocessing
print("Cleaning data...")
# The text fields where red flags usually hide
text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']

# Replace missing values (NaN) with empty strings so we can concatenate safely
for col in text_columns:
    df[col] = df[col].fillna('')

# Combine all text features into one mega-column for the NLP model
df['full_text'] = df['title'] + " " + df['company_profile'] + " " + df['description'] + " " + df['requirements'] + " " + df['benefits']

# Define our features (X) and target/labels (y)
X = df['full_text']
y = df['fraudulent']

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Phase 2: The ML Brain
print("Vectorizing text (TF-IDF)...")
# Convert text into a matrix of TF-IDF features (max 5000 features to keep it fast for V1)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Training Random Forest Classifier...")
# Train the model
# We use class_weight='balanced' because fake jobs are very rare (~5%) compared to real ones
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train_vec, y_train)

# Evaluate the model
print("\n--- Model Evaluation ---")
predictions = model.predict(X_test_vec)

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

import joblib

# Save the trained model and the vectorizer to disk
print("\nSaving model and vectorizer...")
joblib.dump(model, 'fake_job_detector_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Saved successfully! You should see two new .pkl files in your folder.")