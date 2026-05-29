import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack, csr_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

print("Loading dataset...")
df = pd.read_csv('fake_job_postings.csv')

# ── Text columns ───────────────────────────────────────────────────────────────
text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
for col in text_cols:
    df[col] = df[col].fillna('')

df['full_text'] = df[text_cols].apply(lambda r: ' '.join(r), axis=1)

# ── Feature Engineering ────────────────────────────────────────────────────────
def extract_features(df):
    feats = pd.DataFrame()
    text = df['full_text'].str.lower()

    # 1. Missing fields
    feats['missing_company']      = (df['company_profile'] == '').astype(int)
    feats['missing_salary']       = (df['salary_range'].isna() | (df['salary_range'] == '')).astype(int)
    feats['missing_requirements'] = (df['requirements'] == '').astype(int)
    feats['missing_benefits']     = (df['benefits'] == '').astype(int)
    feats['missing_location']     = (df['location'].isna() | (df['location'] == '')).astype(int)

    # 2. Urgency & pressure
    urgency_words = [
        'urgent', 'immediately', 'limited seats', 'apply now', 'hurry',
        '100% guaranteed', 'guaranteed placement', 'instant', 'same day offer'
    ]
    feats['urgency_score'] = text.apply(lambda t: sum(1 for w in urgency_words if w in t))

    # 3. Upfront payment
    payment_words = [
        'registration fee', 'training fee', 'security deposit',
        'processing fee', 'laptop purchase', 'buy your own',
        'refundable deposit', 'pay to join', 'invest to earn'
    ]
    feats['payment_flags'] = text.apply(lambda t: sum(1 for w in payment_words if w in t))

    # 4. Suspicious recruitment process
    process_words = [
        'no interview', 'hired without interview', 'chat interview',
        'whatsapp interview', 'telegram interview', 'offer before assessment',
        'immediate joining', 'join today', 'start tomorrow'
    ]
    feats['bad_process_flags'] = text.apply(lambda t: sum(1 for w in process_words if w in t))

    # 5. Suspicious contact
    contact_words = [
        'whatsapp', 'telegram', 'gmail.com', 'yahoo.com', 'hotmail.com',
        'contact us on', 'message us', 'dm us'
    ]
    feats['suspicious_contact'] = text.apply(lambda t: sum(1 for w in contact_words if w in t))

    # 6. Unrealistic salary
    salary_words = [
        'earn up to', 'earn $', 'earn upto', '5000 per week', '10000 per week',
        'unlimited earning', 'high salary', 'weekly payout', 'daily salary',
        'part time earn', 'work from home earn'
    ]
    feats['salary_flags'] = text.apply(lambda t: sum(1 for w in salary_words if w in t))

    # 7. No experience required
    feats['no_exp_required'] = text.str.contains(
        'no experience|no qualification|fresher welcome|anyone can apply', regex=True
    ).astype(int)

    # 8. Language quality
    feats['exclamation_count'] = df['full_text'].str.count('!')
    feats['caps_ratio'] = df['full_text'].apply(
        lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
    )
    feats['emoji_count'] = df['full_text'].apply(
        lambda t: len(re.findall(r'[^\w\s,.]', t))
    )

    # 9. Description length
    feats['desc_length']    = df['description'].str.len()
    feats['very_short_desc'] = (feats['desc_length'] < 100).astype(int)

    # 10. Structured fields present
    feats['has_employment_type'] = (~df['employment_type'].isna()).astype(int) \
        if 'employment_type' in df.columns else 0
    feats['has_education'] = (~df['required_education'].isna()).astype(int) \
        if 'required_education' in df.columns else 0
    feats['has_experience'] = (~df['required_experience'].isna()).astype(int) \
        if 'required_experience' in df.columns else 0
    feats['has_industry'] = (~df['industry'].isna()).astype(int) \
        if 'industry' in df.columns else 0

    return feats.fillna(0)

print("Engineering features...")
structured_features = extract_features(df)

# ── Train/Test Split ───────────────────────────────────────────────────────────
X_text   = df['full_text']
X_struct = structured_features
y        = df['fraudulent']

(X_text_train, X_text_test,
 X_struct_train, X_struct_test,
 y_train, y_test) = train_test_split(
    X_text, X_struct, y,
    test_size=0.2, random_state=42, stratify=y
)

# ── TF-IDF ────────────────────────────────────────────────────────────────────
print("Vectorizing text...")
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True
)
X_text_train_vec = vectorizer.fit_transform(X_text_train)
X_text_test_vec  = vectorizer.transform(X_text_test)

# ── Scale structured features ─────────────────────────────────────────────────
scaler = StandardScaler()
X_struct_train_scaled = csr_matrix(scaler.fit_transform(X_struct_train))
X_struct_test_scaled  = csr_matrix(scaler.transform(X_struct_test))

# ── Combine ───────────────────────────────────────────────────────────────────
X_train_combined = hstack([X_text_train_vec, X_struct_train_scaled])
X_test_combined  = hstack([X_text_test_vec,  X_struct_test_scaled])

# ── SMOTE — balance the training set ─────────────────────────────────────────
print("Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_combined, y_train)
print(f"After SMOTE — Real: {sum(y_train_resampled==0)}, Fake: {sum(y_train_resampled==1)}")

# ── XGBoost ───────────────────────────────────────────────────────────────────
print("Training XGBoost Classifier...")
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    tree_method='hist',   # fastest CPU method
    n_jobs=-1             # use all cores
)
model.fit(X_train_resampled, y_train_resampled)

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n--- Model Evaluation ---")
predictions = model.predict(X_test_combined)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# ── Save ──────────────────────────────────────────────────────────────────────
print("Saving model artifacts...")
joblib.dump(model,      'fake_job_detector_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(scaler,     'feature_scaler.pkl')
joblib.dump(structured_features.columns.tolist(), 'feature_columns.pkl')
print("Done! Saved: model, vectorizer, scaler, feature_columns")
