from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import io
import os
import re
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
from PIL import Image
import requests
from bs4 import BeautifulSoup
from scipy.sparse import hstack, csr_matrix

# ── Load model artifacts ───────────────────────────────────────────────────────
print("Loading model artifacts...")
model           = joblib.load('fake_job_detector_model.pkl')
vectorizer      = joblib.load('tfidf_vectorizer.pkl')
scaler          = joblib.load('feature_scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Fake Job Detector API v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ────────────────────────────────────────────────────────────────────
class JobData(BaseModel):
    text: str

class URLData(BaseModel):
    url: str

# ── Red flag keyword categories (shown to user) ────────────────────────────────
FLAG_CATEGORIES = {
    "💸 Upfront Payment": [
        "registration fee", "training fee", "security deposit",
        "processing fee", "laptop purchase", "pay to join", "invest to earn",
        "refundable deposit"
    ],
    "📱 Suspicious Contact": [
        "whatsapp", "telegram", "gmail.com", "yahoo.com",
        "hotmail.com", "message us on", "dm us"
    ],
    "🚨 Urgency / Pressure": [
        "urgent", "limited seats", "apply immediately", "hurry",
        "100% guaranteed", "guaranteed placement", "instant offer",
        "same day offer"
    ],
    "🎭 Suspicious Process": [
        "no interview", "hired without interview", "chat interview",
        "whatsapp interview", "offer before assessment",
        "immediate joining", "join today", "start tomorrow"
    ],
    "💰 Unrealistic Salary": [
        "earn up to", "earn $", "5000 per week", "10000 per week",
        "unlimited earning", "weekly payout", "daily salary",
        "part time earn", "work from home earn"
    ],
    "⚠️ Vague Requirements": [
        "no experience required", "no qualification", "anyone can apply",
        "fresher welcome"
    ]
}

def scan_flags(text: str) -> dict:
    """Returns dict of category -> list of matched phrases."""
    text_lower = text.lower()
    found = {}
    for category, keywords in FLAG_CATEGORIES.items():
        matches = [kw for kw in keywords if kw in text_lower]
        if matches:
            found[category] = matches
    return found

# ── Structured feature extraction (mirrors train_model.py) ────────────────────
def extract_structured_features(text: str) -> pd.DataFrame:
    t = text.lower()

    urgency_words = [
        'urgent', 'immediately', 'limited seats', 'apply now', 'hurry',
        '100% guaranteed', 'guaranteed placement', 'instant', 'same day offer'
    ]
    payment_words = [
        'registration fee', 'training fee', 'security deposit',
        'processing fee', 'laptop purchase', 'buy your own',
        'refundable deposit', 'pay to join', 'invest to earn'
    ]
    process_words = [
        'no interview', 'hired without interview', 'chat interview',
        'whatsapp interview', 'telegram interview', 'offer before assessment',
        'immediate joining', 'join today', 'start tomorrow'
    ]
    contact_words = [
        'whatsapp', 'telegram', 'gmail.com', 'yahoo.com', 'hotmail.com',
        'contact us on', 'message us', 'dm us'
    ]
    salary_words = [
        'earn up to', 'earn $', 'earn upto', '5000 per week',
        '10000 per week', 'unlimited earning', 'high salary',
        'weekly payout', 'daily salary', 'part time earn', 'work from home earn'
    ]

    row = {
        'missing_company':      0,   # can't detect from raw text alone
        'missing_salary':       int('salary' not in t and '₹' not in t and '$' not in t),
        'missing_requirements': int(len(t) < 200),
        'missing_benefits':     int('benefit' not in t and 'insurance' not in t),
        'missing_location':     int('location' not in t and 'remote' not in t and 'office' not in t),
        'urgency_score':        sum(1 for w in urgency_words if w in t),
        'payment_flags':        sum(1 for w in payment_words if w in t),
        'bad_process_flags':    sum(1 for w in process_words if w in t),
        'suspicious_contact':   sum(1 for w in contact_words if w in t),
        'salary_flags':         sum(1 for w in salary_words if w in t),
        'no_exp_required':      int(bool(re.search(r'no experience|no qualification|fresher welcome|anyone can apply', t))),
        'exclamation_count':    text.count('!'),
        'caps_ratio':           sum(1 for c in text if c.isupper()) / max(len(text), 1),
        'emoji_count':          len(re.findall(r'[^\w\s,.]', text)),
        'desc_length':          len(text),
        'very_short_desc':      int(len(text) < 100),
        'has_employment_type':  int(bool(re.search(r'full.time|part.time|contract|freelance', t))),
        'has_education':        int(bool(re.search(r'bachelor|master|degree|diploma|graduate', t))),
        'has_experience':       int(bool(re.search(r'\d+\s*year|experience required|yrs', t))),
        'has_industry':         int(bool(re.search(r'industry|sector|domain', t))),
    }

    df = pd.DataFrame([row])
    # Ensure columns match training order exactly
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]

# ── Core prediction ────────────────────────────────────────────────────────────
def get_prediction(text: str):
    # 1. TF-IDF
    text_vec = vectorizer.transform([text])

    # 2. Structured features
    struct_df = extract_structured_features(text)
    struct_scaled = csr_matrix(scaler.transform(struct_df))

    # 3. Combine and predict
    combined = hstack([text_vec, struct_scaled])
    prediction = model.predict(combined)
    probabilities = model.predict_proba(combined)

    ml_says_fake = bool(prediction[0] == 1)
    scam_prob = round(probabilities[0][1] * 100, 2)

    # 4. Keyword flags (for UI display)
    found_flags = scan_flags(text)
    has_flags = len(found_flags) > 0

    # 5. Override: if hard flags found, ensure it's marked fake
    is_fake = ml_says_fake or has_flags
    if has_flags and scam_prob < 65:
        scam_prob = 65.0

    # Flatten flags for JSON response
    flat_flags = []
    for category, matches in found_flags.items():
        for match in matches:
            flat_flags.append(f'{category}: "{match}"')

    return {
        "is_fake": is_fake,
        "scam_probability": scam_prob,
        "flags": flat_flags,
        "status": "success"
    }

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.post("/predict")
def predict_job(job: JobData):
    return get_prediction(job.text)

@app.post("/predict_url")
def predict_url(data: URLData):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(data.url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        if not text:
            return {"status": "error", "message": "Could not extract text from that URL."}
        return get_prediction(text)
    except Exception:
        return {"status": "error", "message": "Failed to fetch that URL."}

@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    content = await file.read()
    text = ""
    try:
        filename = file.filename.lower()
        if filename.endswith(".pdf"):
            pdf = PdfReader(io.BytesIO(content))
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
        elif filename.endswith((".png", ".jpg", ".jpeg")):
            return {"status": "error", "message": "Image upload not supported on cloud. Use PDF instead."}
        else:
            return {"status": "error", "message": "Unsupported file type."}

        if not text.strip():
            return {"status": "error", "message": "Could not extract text from this file."}
        return get_prediction(text)
    except Exception as e:
        return {"status": "error", "message": str(e)}
