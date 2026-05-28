from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import io
import os
from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
import requests
from bs4 import BeautifulSoup

# ── Load ML model ──────────────────────────────────────────────────────────────
print("Loading model and vectorizer...")
model = joblib.load('fake_job_detector_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Fake Job Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Tesseract path — auto-detects Windows vs Mac/Linux ────────────────────────
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ── Scam keyword list ──────────────────────────────────────────────────────────
SCAM_KEYWORDS = [
    "urgent hiring",
    "bank details",
    "processing fee",
    "registration fee",
    "whatsapp",
    "no experience required",
    "telegram",
    "cashapp",
    "wire transfer",
    "guaranteed income",
    "earn $",
    "earn up to",
    "work from home and earn",
    "send us your details",
    "immediate joining",
    "no interview",
    "payment required",
    "part time earn",
    "weekly salary",
    "daily salary",
]

# ── Schemas ────────────────────────────────────────────────────────────────────
class JobData(BaseModel):
    text: str

class URLData(BaseModel):
    url: str

# ── Core prediction logic (used by all 3 endpoints) ───────────────────────────
def get_prediction(text: str):
    # 1. ML model prediction
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    probabilities = model.predict_proba(text_vectorized)

    ml_says_fake = bool(prediction[0] == 1)
    scam_prob = round(probabilities[0][1] * 100, 2)

    # 2. Keyword scan
    text_lower = text.lower()
    found_flags = [kw for kw in SCAM_KEYWORDS if kw in text_lower]
    has_flags = len(found_flags) > 0

    # 3. Final verdict: fake if ML says so OR keywords found
    #    If keywords found but ML probability is low, bump to at least 65%
    is_fake = ml_says_fake or has_flags
    if has_flags and scam_prob < 65:
        scam_prob = 65.0

    return {
        "is_fake": is_fake,
        "scam_probability": scam_prob,
        "flags": found_flags,
        "status": "success"
    }

# ── Endpoint 1: Raw text (auto-scan & manual paste) ───────────────────────────
@app.post("/predict")
def predict_job(job: JobData):
    return get_prediction(job.text)

# ── Endpoint 2: URL scraper ────────────────────────────────────────────────────
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
    except Exception as e:
        return {"status": "error", "message": "Failed to fetch that URL. It may be blocked or invalid."}

# ── Endpoint 3: File upload (PDF or image) ────────────────────────────────────
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
            image = Image.open(io.BytesIO(content))
            text = pytesseract.image_to_string(image)

        else:
            return {"status": "error", "message": "Unsupported file type. Please upload a PDF, PNG, or JPG."}

        if not text.strip():
            return {"status": "error", "message": "Could not extract any text from this file."}

        return get_prediction(text)

    except Exception as e:
        return {"status": "error", "message": str(e)}
