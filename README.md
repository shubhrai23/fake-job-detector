# 🔍 Fake Job Detector

An AI-powered Chrome Extension that detects fraudulent job postings in real time using DistilBERT and NLP — built specifically for Indian job seekers on platforms like Naukri, Internshala, and WhatsApp job forwards.

## 🚀 Live Demo
**API:** https://shubhrai23-fake-job-detector.hf.space/docs

---

## 🧠 How It Works

1. User opens a job posting on Naukri, Internshala, Glassdoor, Indeed, or LinkedIn
2. The extension automatically extracts only the job description (not the whole page)
3. Text is sent to a FastAPI backend hosted on Hugging Face Spaces
4. A fine-tuned DistilBERT transformer model returns a scam probability score
5. A keyword scanner checks for red flags (WhatsApp contact, processing fees, urgency language, etc.)
6. Result is displayed instantly in the popup with flagged phrases listed

---

## ✨ Features

- **Zero-click auto-scan** — results appear the moment you open the extension
- **Smart DOM extractor** — targets job panels specifically on Naukri, Internshala, Indeed, LinkedIn, Glassdoor
- **PDF upload** — scan job descriptions sent as PDF files
- **Red flag explanations** — shows exactly which phrases triggered the alert
- **Live cloud API** — works for anyone, no local setup needed

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | DistilBERT (fine-tuned transformer) |
| Training | PyTorch + Hugging Face Transformers |
| Backend API | FastAPI + Uvicorn |
| PDF Parsing | PyPDF2 |
| Web Scraping | BeautifulSoup + Requests |
| Frontend | Chrome Extension (Vanilla JS) |
| Deployment | Hugging Face Spaces (Docker) |
| Dataset | EMSCAD — 18,000 labeled job postings |

---

## 📊 Model Performance

| Version | Model | Recall (fake) | Precision (fake) |
|---|---|---|---|
| V1 | Random Forest + TF-IDF | 0.58 | 0.99 |
| V2 | Gradient Boosting + structured features | 0.72 | 0.92 |
| V3 | XGBoost + SMOTE | 0.73 | 0.90 |
| **V4** | **DistilBERT (fine-tuned)** | **0.87** | **0.91** |

- **Dataset:** Employment Scam Aegean Dataset (EMSCAD) — ~18,000 real and fake job postings
- **Training:** Google Colab T4 GPU, 4 epochs, class-weighted loss

---

## 🎯 Target Use Cases

This tool is designed for:
- **Naukri / Internshala / Shine** — Indian job boards with weaker moderation
- **WhatsApp / Telegram job forwards** — copy-paste the job text into any page and scan
- **Unknown company career pages** — fake sites designed to harvest resumes and charge fees

> LinkedIn and Indeed have strong built-in moderation. This tool fills the gap for platforms that don't.

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

> The extension connects to the live Hugging Face API — no local server needed.
> First request may take ~30 seconds if the API is waking from sleep.

---

## 📁 Project Structure

```
fake-job-detector/
├── api.py                        # FastAPI backend (DistilBERT)
├── train_model.py                # Model training script
├── requirements.txt              # Python dependencies
├── Dockerfile                    # HF Spaces deployment config
├── config.json                   # DistilBERT model config
├── model.safetensors             # Fine-tuned model weights
├── tokenizer.json                # Tokenizer
└── extension/
    ├── manifest.json             # Chrome extension config
    ├── popup.html                # Extension UI
    └── popup.js                  # Smart scraper + API calls
```

---

## 💡 Resume Story

> Fine-tuned DistilBERT on 18,000+ job postings achieving 87% recall on fraudulent listings. Built a FastAPI backend with PDF parsing and a smart DOM scraper targeting job panels on Naukri, Internshala, Indeed, and LinkedIn. Deployed on Hugging Face Spaces as a live Chrome Extension with real-time scam probability scoring and red flag detection — targeting Indian job seekers on platforms with weak moderation.