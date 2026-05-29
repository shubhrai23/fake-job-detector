# 🔍 Fake Job Detector

An AI-powered Chrome Extension that detects fraudulent job postings in real time using NLP and machine learning.

## 🚀 Live Demo
**API:** https://fake-job-detector-production-31b1.up.railway.app/docs

---

## 🧠 How It Works

1. User opens a job posting on Indeed, LinkedIn, Naukri, Internshala, or Glassdoor
2. The extension automatically extracts only the job description (not the whole page)
3. Text is sent to a FastAPI backend hosted on Railway
4. A trained Random Forest + TF-IDF classifier returns a scam probability score
5. A keyword scanner checks for hardcoded red flags (WhatsApp, processing fee, etc.)
6. Result is displayed instantly in the popup with flagged phrases listed

---

## ✨ Features

- **Zero-click auto-scan** — results appear the moment you open the extension
- **Smart DOM extractor** — targets job panels specifically on Indeed, LinkedIn, Naukri, Internshala, Glassdoor instead of scraping the whole page
- **PDF upload** — scan job descriptions sent as PDF files
- **URL scanner** — paste any job link and scan it directly
- **Red flag explanations** — shows exactly which phrases triggered the alert
- **Live cloud API** — works for anyone, no local setup needed

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Scikit-learn (Random Forest + TF-IDF) |
| Backend API | FastAPI + Uvicorn |
| PDF Parsing | PyPDF2 |
| Web Scraping | BeautifulSoup + Requests |
| Frontend | Chrome Extension (Vanilla JS) |
| Deployment | Railway |
| Dataset | EMSCAD — 18,000 labeled job postings |

---

## 📊 Model Performance

- **Dataset:** Employment Scam Aegean Dataset (EMSCAD) — ~18,000 real and fake job postings
- **Precision (fraudulent class):** 0.99
- **Recall (fraudulent class):** 0.58
- **Approach:** TF-IDF vectorization (5000 features) + class-balanced Random Forest

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict` | Classify raw text |
| POST | `/predict_url` | Scrape and classify a URL |
| POST | `/predict_file` | Upload and classify a PDF |

---

## 🧩 Install the Extension (Local)

1. Clone this repo
2. Open Chrome → `chrome://extensions/`
3. Enable **Developer Mode** (top right)
4. Click **Load unpacked**
5. Select the `extension/` folder
6. Pin the extension and visit any job board

> The extension connects to the live Railway API — no local server needed.

---

## 📁 Project Structure

```
fake-job-detector/
├── api.py                        # FastAPI backend
├── train_model.py                # Model training script
├── requirements.txt              # Python dependencies
├── Procfile                      # Railway deployment config
├── fake_job_detector_model.pkl   # Trained ML model
├── tfidf_vectorizer.pkl          # Fitted TF-IDF vectorizer
└── extension/
    ├── manifest.json             # Chrome extension config
    ├── popup.html                # Extension UI
    └── popup.js                  # Smart scraper + API calls
```

---

## 💡 Resume Story

> Trained an NLP classifier on 18,000+ job postings to detect fraudulent listings. Built a FastAPI backend with PDF parsing and a smart DOM scraper targeting job panels on Indeed, LinkedIn, Naukri, and Internshala. Deployed as a live Chrome Extension with real-time scam probability scoring and red flag explanations.
