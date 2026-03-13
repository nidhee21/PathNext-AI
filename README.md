# 🧭 PathNext AI — Connecting Skills and Future

> An AI-powered career recommendation system for students after 12th grade.
> Built with Flask + FastAPI + Scikit-learn. Pure HTML/CSS/JS frontend.

---

## 📋 Project Overview

PathNext AI helps students discover their ideal career path by analyzing:
- **Academic performance** (stream + subject marks)
- **Personal interests** (30 options, max 10 selectable)
- **Soft skills** (rated 1–5)

The system predicts **Top 3 career paths** with confidence scores using a
Random Forest ML model, then shows skills required, a beginner roadmap,
and top colleges in Mumbai for each career.

---

## 🏗 Architecture

```
Frontend (HTML/CSS/JS)
    │
    │ POST /predict
    ▼
Flask Server (port 5000)         ← serves HTML pages + handles form
    │
    │ POST /predict (internal call)
    ▼
FastAPI Server (port 8001)       ← ML prediction API
    │
    │ loads
    ▼
career_model.pkl                 ← trained Random Forest model
```

---

## 📁 Project Structure

```
pathnext-ai/
│
├── frontend/
│   ├── index.html          ← Landing page
│   ├── assessment.html     ← Assessment form
│   ├── results.html        ← Results with charts
│   ├── css/
│   │   └── styles.css      ← All styles (dark glassmorphism)
│   └── js/
│       ├── animations.js   ← Scroll + typing animations
│       └── charts.js       ← Chart.js bar/radar/pie charts
│
├── backend/
│   ├── flask_app.py        ← Flask server (serves frontend + routes)
│   ├── fastapi_app.py      ← FastAPI ML prediction API
│   └── ml_model.py         ← Model training script
│
├── dataset/
│   └── career_dataset.csv  ← Generated training data (3000 rows)
│
├── models/
│   └── career_model.pkl    ← Trained model (generated after training)
│
├── scripts/
│   └── generate_dataset.py ← Dataset generation script
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### Step 1 — Install Python packages
```bash
pip install -r requirements.txt
```

### Step 2 — Generate training dataset
```bash
cd scripts
python generate_dataset.py
```

### Step 3 — Train the ML model
```bash
cd backend
python ml_model.py
```
You should see: `Random Forest Accuracy: ~97%`

### Step 4 — Start FastAPI (Terminal 1)
```bash
cd backend
uvicorn fastapi_app:app --port 8001 --reload
```
FastAPI runs at: http://localhost:8001
API docs at:     http://localhost:8001/docs

### Step 5 — Start Flask (Terminal 2)
```bash
cd backend
python flask_app.py
```
Website runs at: http://localhost:5000

### Step 6 — Open browser
Go to: **http://localhost:5000**

---

## 🌐 Works Without Backend Too!

The frontend HTML files work as standalone files.
Just double-click `frontend/index.html` to open it in your browser.
The assessment and results pages use mock ML predictions when Flask is not running.

---

## 🧠 How the ML Model Works

1. **Dataset**: 3000 synthetic student profiles generated with realistic rules
   - Science PCM students: biology = 0
   - Science PCB students: maths = 0
   - Career labels match subject/interest patterns

2. **Features used**:
   - Stream (encoded as number)
   - Subject marks (0–100)
   - Interest flags (30 binary columns)
   - Soft skill ratings (10 columns, 1–5)

3. **Model**: Random Forest Classifier (100 trees)
   - Trained on 80% of data
   - Tested on 20% → ~97% accuracy

4. **Prediction**: `predict_proba()` returns probability for all 15 careers.
   Top 3 are returned with confidence scores.

---

## 🎨 Design

- Dark background with neon blue and purple accents
- Glassmorphism cards with blur effect
- Glow animations on hover
- Floating blob shapes in hero section
- Smooth scroll fade-in animations
- Typing animation for AI explanation
- Animated Chart.js charts on results page

---

## 📚 Careers Covered

Software Engineer, Data Scientist, Cybersecurity Analyst,
Doctor, Biotechnologist, Chartered Accountant, Investment Banker,
Entrepreneur, Lawyer, Journalist, UX Designer,
Psychologist, Architect, Digital Marketer, Teacher

---

## 🔮 Future Improvements

- [ ] Add real OpenAI API for dynamic explanations
- [ ] User authentication and saved results
- [ ] More careers (50+)
- [ ] Real student data for better accuracy
- [ ] PDF export of results
- [ ] Aptitude test questions
- [ ] Hindi language support

---

Made with ❤️ for students figuring out their future.
