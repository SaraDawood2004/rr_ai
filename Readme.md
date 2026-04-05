# ReviewRanker — AI-Powered Review Helpfulness Predictor

        > **Predict how helpful a product review is — before anyone votes on it.**   <

---------------------------------------------------------------------------------------------------------------

## 1. Project Title & Overview

**ReviewRanker: An NLP-Based Product Review Helpfulness Prediction System**

ReviewRanker is an end-to-end machine learning system that automatically predicts the helpfulness of product reviews using Natural Language Processing (NLP). It analyzes a review's text quality, grammar richness, and user behaviour signals to assign a helpfulness score between 0 and 100%, and tags the review as:

| Color      |       Tag         |     Score Range |
|------------|-------------------|-----------------|
| 🟢 Green  |   **Useful**       |      ≥ 75%      |
| 🟡 Yellow | **Moderately Useful** |    40–74%    |
| 🔴 Red | **Not Useful**        |          < 40% |

The system works as a **Chrome browser plugin** that automatically activates on popular 
e-commerce sites (Amazon, Myntra), injecting helpfulness score badges directly onto each review — no user action needed.

---------------------------------------------------------------------------------------------------------------

## 2. Problem Statement

When customers shop online, they rely heavily on product reviews to make purchase decisions. However:

- **Volume overload** — popular products can have thousands or even millions of reviews
- **Quality imbalance** — not all reviews are equally informative; many are vague, emotionally driven, or spam
- **Time waste** — users spend significant time reading unhelpful reviews like *"Great product!! Love it!!!"* which provide no actionable information
- **Delayed helpfulness signal** — the traditional helpfulness rating (👍/👎 voting) only becomes meaningful after many users have voted, making it useless for new reviews
- **No pre-screening** — current systems have no mechanism to predict whether a review will be helpful *before* it goes live

**Core question:** *Can we predict whether a product review is genuinely helpful — before other users vote on it?*

---------------------------------------------------------------------------------------------------------------

## 3. Existing Solutions & Limitations

### 3.1 Amazon's Helpful Votes System
Amazon allows users to mark reviews as helpful. Reviews with more votes rise to the top.

**Limitations:**
- Requires a large number of votes to be meaningful — new reviews are invisible
- Susceptible to vote manipulation and coordinated fake voting
- Provides no pre-screening; unhelpful reviews appear for weeks before being pushed down
- Purely reactive — does not analyze review content at all

### 3.2 Star Rating Filters
Users can filter reviews by 1-star or 5-star ratings to find extreme opinions.

**Limitations:**
- Star rating does not measure content quality or informativeness
- A 5-star review can still be completely useless ("Amazing!! 10/10")
- Does not detect spam or repetitive reviews

### 3.3 Sentiment Analysis Tools
Some third-party tools (Fakespot, ReviewMeta) analyze review sentiment and flag suspicious patterns.

**Limitations:**
- Focused on fake review detection, not helpfulness prediction
- Require browser extensions with proprietary backends — not open source
- Do not analyze linguistic quality (grammar diversity, readability, content depth)
- Do not inject scores directly into the shopping page UI in real time

### 3.4 Summary

|            Feature           | Amazon Votes | Star Filter | Fakespot | **ReviewRanker** |
|------------------------------|--------------|-------------|----------|------------------|
| Pre-vote prediction          |        ✗     |     ✗      |      ✗   |      ✅         |
| Content quality analysis     |        ✗     |     ✗      |   Partial|       ✅        |
| Grammar analysis             |        ✗     |     ✗      |      ✗   |      ✅         |
| Spam detection               |        ✗     |     ✗      |      ✗   |      ✅         |
| Works on multiple sites      |        N/A   |     N/A    |    Limited |       ✅        |
| In-page badge injection      |        ✗     |     ✗      |      ✗   |      ✅         |
| Open source / free           |        ✗     |     ✗      |      ✗   |      ✅         |

---------------------------------------------------------------------------------------------------------------

## 4. Uniqueness of Proposed Approach

ReviewRanker introduces a **3-component feature engineering pipeline** that analyzes reviews from three orthogonal perspectives simultaneously:

### Component 1 — Text Structure Analysis
Analyzes *what* the reviewer said:
- **Sentiment polarity** using VADER (compound, positive, negative, neutral scores)
- **Readability** using Flesch Reading Ease and Flesch-Kincaid Grade Level
- **Content depth**: word count, sentence count, average word/sentence length
- **Product feature mentions**: detects specific product attributes (quality, price, taste, battery, etc.)
- **Opinion balance**: detects whether the reviewer mentions both pros AND cons — balanced reviews are more credible

### Component 2 — Grammar Structure Analysis
Analyzes *how* the reviewer wrote:
- **POS (Part-of-Speech) tag diversity** using spaCy — a review with varied nouns, verbs, adjectives, adverbs is linguistically richer
- **Lexical diversity** (Type-Token Ratio) — measures vocabulary richness
- **Emotion intensity**: counts strong emotion words ("absolutely", "genuinely", "despise")
- **Writing quality signals**: ALL-CAPS ratio, excessive punctuation (!!!), presence of numbers (specific data = more credible)

### Component 3 — User Activity Signals
Analyzes *who* wrote the review:
- **Reviewer history**: number of past reviews, average star rating given
- **Rating deviation**: how far this review's star rating deviates from the product average — extreme outliers are suspicious
- **Spam detection**: flags reviewers who post at high frequency with zero rating variation
- **Product-level context**: computes product average and reviewer-vs-product alignment

### What Makes It Unique
1. **Predictive, not reactive** — scores reviews at submission time, before any votes exist
2. **3-component NLP pipeline** — most existing tools use only sentiment; this system combines text quality + grammar + behavioral signals
3. **Live page integration** — as a Chrome plugin, it injects color-coded badges directly into the shopping page UI
4. **Multi-site** — works across Amazon, Flipkart, Walmart, Best Buy, eBay, Myntra
5. **Fully open source and deployable for free**

---------------------------------------------------------------------------------------------------------------

## 5. System Architecture

```
User opens product page
        │
        ▼
  content.js (plugin)
  scrapes reviews + user signals from DOM
        │
        ▼
  POST /api/predict-bulk
  (Flask API on localhost:5000)
        │
        ▼
  predict.py
  → features.py (3-component extraction)
  → ML model (Random Forest)
        │
        ▼
  Returns: score + tag + breakdown
        │
        ▼
  content.js injects badges
  Reviews reordered by score (best first)
```

--------------------------------------------------------------------------------------------------------------

## 6. Project Structure

```
ReviewRanker/
│
├── app/                          # Flask web application
│   ├── app.py                    # REST API (predict + predict-bulk endpoints)
│   └── templates/
│       └── index.html            # Web UI (dark violet theme)
│
├── src/                          # Core ML pipeline
│   ├── preprocess.py             # Data cleaning + sampling
│   ├── features.py               # 3-component NLP feature extraction
│   ├── train_model.py            # Model training + evaluation
│   └── predict.py                # Prediction logic + tag assignment
│
├── models/                       # Saved trained artifacts
│   ├── helpfulness_model.pkl     # Trained Random Forest model
│   └── feature_cols.pkl          # Feature column names (for inference)
│
├── data/                         # Dataset files
│   ├── Reviews.csv               # Raw Amazon Fine Food Reviews (~568K rows)
│   ├── reviews_sample.csv        # Cleaned 10K stratified sample
│   └── reviews_features.csv      # Feature-engineered dataset (ready for training)
│
├── plugin/                       # Chrome browser extension
│   ├── manifest.json             # Extension config + permissions
│   ├── content.js                # Page scraper + badge injector (runs on e-commerce sites)
│   ├── popup.html                # Plugin popup UI
│   ├── popup.css                 # Popup styles
│   ├── popup.js                  # Popup logic (CSP-compliant, no inline JS)
│   └── logo.png                  # Extension icon
│
├── render_build.sh               # Build script for Render deployment
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---------------------------------------------------------------------------------------------------------------

## 7. File-by-File Code Summary

### `src/preprocess.py`
**Purpose:** Cleans and samples the raw Amazon dataset into a usable 10K training set.

**What it does — step by step:**
1. **Loads** the raw `Reviews.csv` (~568K rows) and renames columns to clean snake_case names
2. **Filters** rows where total helpfulness votes < 5 — a review with 1/1 votes is technically 100% helpful but statistically meaningless
3. **Creates the target label**: `helpfulness_score = helpful_votes / total_votes` — a float between 0.0 and 1.0. This is what the ML model will learn to predict
4. **Cleans text** — strips HTML tags, URLs, special characters; *intentionally keeps* `! ? . ,` because they carry emotion signals
5. **Removes short reviews** (< 20 words) — "Great product!" has no NLP features to extract
6. **Removes duplicates** — some users post identical reviews on multiple products
7. **Stratified sampling** — splits reviews into low/mid/high score buckets and samples equally (3,333 from each), preventing the model from being biased toward one type

**Key output:** `reviews_sample.csv` — 10,000 balanced, clean reviews

---

### `src/features.py`
**Purpose:** Extracts ~40 NLP features from each review across 3 components.

**Component 1 — `extract_text_structure(text, summary)`**
- Runs VADER sentiment analysis → gives compound/positive/negative/neutral scores
- Runs `textstat` → Flesch Reading Ease (0–100, higher = easier to read)
- Counts words, sentences, characters, average word/sentence length
- Scans for product feature keywords (quality, price, taste, battery, etc.)
- Counts pros keywords (love, excellent, amazing) and cons keywords (terrible, broken, waste)
- Flags if review has BOTH pros and cons → balanced opinion = more helpful
- Compares title and body sentiment → coherence score

**Component 2 — `extract_grammar_structure(text)`**
- Uses spaCy to POS-tag every word → calculates noun/verb/adjective/adverb ratios
- Computes POS diversity (unique POS types / 17 total) → richer grammar = more helpful
- Counts emotion intensity words (absolutely, deeply, genuinely)
- Measures ALL-CAPS ratio and excessive `!` → writing quality signals
- Calculates lexical diversity (unique words / total words) → vocabulary richness
- Detects numbers in text → specific data like "used for 3 months, paid $25" adds credibility

**Component 3 — `build_user_stats(df)`**
- Groups by `user_id` → computes total review count, average star rating, std deviation
- Computes `reviews_per_day` → high frequency reviewers are suspicious
- Flags `is_suspicious = 1` if posting > 3 reviews/day AND always gives same rating
- Groups by `product_id` → computes product average score
- Computes `score_deviation = |this review's stars − product average|` → outlier detection

**Key output:** `reviews_features.csv` — 10,000 rows × 51 columns

---

### `src/train_model.py`
**Purpose:** Trains two ML models, compares them, and saves the best one.

**What it does:**
1. Loads `reviews_features.csv` and selects the 41 feature columns
2. Splits data 80% train / 20% test with stratified splits (balanced score buckets)
3. Trains **Random Forest Regressor** (200 trees, max depth 12) — robust to noisy features
4. Trains **XGBoost Regressor** (300 trees, learning rate 0.05) — gradient boosting
5. Evaluates both on RMSE, MAE, R² score
6. Saves the **best model** (lower RMSE wins) as `models/helpfulness_model.pkl`
7. Saves feature column names as `models/feature_cols.pkl` — needed at inference time
8. Plots top 15 feature importances → `models/feature_importance.png`

**Results:**
- Random Forest: RMSE 0.2552, R² 0.36 ✅ (winner)
- XGBoost: RMSE 0.2562, R² 0.354

> The R² of ~0.36 is expected and reasonable for this task — human helpfulness perception is subjective and noisy. The model's relative ranking of reviews is far more accurate than the absolute score.

---

### `src/predict.py`
**Purpose:** Loads the trained model and predicts helpfulness for any new review text.

**What it does:**
1. Loads `helpfulness_model.pkl` and `feature_cols.pkl` at import time (once)
2. `predict_review(text, summary, user_context)` — main prediction function:
   - Runs Component 1 + 2 feature extraction on the review text
   - Merges with user context (Component 3 signals)
   - Builds a single-row DataFrame in correct feature order
   - Runs model prediction → clamps result to [0.0, 1.0]
3. `assign_tag(score)` → maps score to tag:
   - ≥ 0.75 → **Useful** (green)
   - 0.40–0.74 → **Moderately Useful** (yellow)
   - < 0.40 → **Not Useful** (red)
4. Returns full result dict including score, percent, tag, color, and 3-component breakdown

**Used by:** `app/app.py` (Flask API endpoints)

---

### `app/app.py`
**Purpose:** Flask REST API — the bridge between the ML backend and the plugin/web UI.

**Endpoints:**

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/`      | Serves the web UI (`index.html`) |
| POST | `/api/predict` | Analyzes a single review → returns score + tag + breakdown |
| POST | `/api/predict-bulk` | Analyzes a list of reviews → returns them sorted by score |

**Key behaviors:**
- `CORS` is enabled — this allows the Chrome plugin (running on amazon.com) to call localhost:5000 without browser security errors
- `/api/predict-bulk` calls `predict_review()` for each review, then sorts results by `score` descending — so the most helpful review is always first
- Both endpoints accept optional `user_context` dict so the plugin can pass scraped star ratings, review counts, etc.

---

### `plugin/manifest.json`
**Purpose:** Chrome Extension configuration — tells Chrome what the plugin can do and where it runs.

**Key settings:**
- `manifest_version: 3` — required for modern Chrome extensions
- `content_scripts` — injects `content.js` automatically on Amazon, Flipkart, Walmart, Best Buy, eBay, Myntra
- `host_permissions` — allows the plugin to make API calls to `localhost:5000`
- `action` → `popup.html` — defines what appears when user clicks the extension icon

---

### `plugin/content.js`
**Purpose:** The core plugin engine — runs invisibly inside every supported e-commerce page.

**What it does:**
1. **Site detection** — auto-detects which site you're on using `SITE_ADAPTERS` (each adapter has CSS selectors for that site's review structure)
2. **Scrapes reviews** — finds all `[data-hook="review"]` elements (Amazon) or equivalent, extracts review text, title, star rating, helpful vote count
3. **Computes user signals** — derives `product_avg_score` and `score_deviation` from all visible reviews on the page; detects suspicious reviewers
4. **Sends to API** — `POST /api/predict-bulk` with all reviews + user context
5. **Injects badges** — adds a colored pill badge `● Helpfulness Score: 82%` under each review's star rating
6. **Re-orders reviews** — moves DOM elements so highest-scored review appears first
7. **MutationObserver** — watches for new reviews loaded by scroll/pagination, and automatically scores them too
8. **`window.__rrReanalyze()`** — exposed globally so the popup's "Re-analyze" button can trigger a fresh run

---

### `plugin/popup.html` + `popup.css` + `popup.js`
**Purpose:** The UI shown when the user clicks the ReviewRanker icon in Chrome's toolbar.

**Why 3 separate files?**
Chrome Extensions enforce **Content Security Policy (CSP)** — inline `<script>` and `onclick=` are forbidden. All JS must be in an external `.js` file and all CSS in an external `.css` file.

**What the popup does:**
- **Server status check** — auto-pings Flask on load; shows green dot if running, red dot if offline
- **"Check Helpfulness Score" button** — user pastes any review text, clicks the button, gets instant score with breakdown (useful for checking a review *before* posting it anywhere)
- **"Re-analyze All Reviews on Page" button** — clears existing badges and re-runs fresh analysis on all reviews currently visible on the tab
- **Result card** — shows colored dot, tag label, percentage score, progress bar, and 6-field breakdown (sentiment, readability, word count, feature mentions, balanced opinion, suspicious flag)

---

### `render_build.sh`
**Purpose:** Automated build script for deploying to Render (free hosting).

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Render runs this script before starting the server. It installs all Python packages and downloads the spaCy English language model.

---

### `requirements.txt`
All Python packages the project depends on:

```
flask
flask-cors
pandas
numpy
scikit-learn
xgboost
nltk
spacy
textstat
vaderSentiment
joblib
matplotlib
seaborn
gunicorn
```

> `gunicorn` is the production-grade WSGI server used when deploying to Render instead of Flask's built-in development server.

---------------------------------------------------------------------------------------------------------------

## 8. How to Run the Project

### Prerequisites
- Python 3.9+
- Google Chrome browser
- Kaggle account (to download the dataset)

### Step 1 — Clone and install dependencies
```bash
git clone https://github.com/SaraDawood2004/reviewrankerai.git
cd ReviewRanker
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 2 — Download the dataset
1. Go to: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
2. Download and extract → you get `Reviews.csv`
3. Move it to: `data/Reviews.csv`

### Step 3 — Run the full ML pipeline
```bash
# From the project root
python src/preprocess.py       # Creates data/reviews_sample.csv
python src/features.py         # Creates data/reviews_features.csv (~5 min)
python src/train_model.py      # Trains model, saves to models/
python src/predict.py          # Quick test — prints 3 sample predictions
```

### Step 4 — Start the Flask API
```bash
python app/app.py
# Server starts at: http://localhost:5000
```

### Step 5 — Open the Web UI
Navigate to `http://localhost:5000` in your browser.

---

## 9. How to Install as a Chrome Plugin

> Flask must be running on port 5000 before using the plugin.

**Step 1 — Generate the icon** - Plugin folder must have icon.png
```python
# Run this once if you don't have logo.png
from PIL import Image
img = Image.new('RGB', (48, 48), color='#7c3aed')
img.save('plugin/logo.png')
```

**Step 2 — Open Chrome Extensions page**
Type in the address bar:
```
chrome://extensions
```

**Step 3 — Enable Developer Mode**
Toggle the **Developer mode** switch in the top-right corner → it turns blue.

**Step 4 — Load the plugin**
1. Click **"Load unpacked"** button (appears after enabling Developer mode)
2. In the file picker, navigate to your project folder
3. Select the **`plugin/`** folder (not any file inside it — the folder itself)
4. Click **"Select Folder"**

**Step 5 — Pin the extension**
1. Click the puzzle piece icon (🧩) in Chrome's toolbar
2. Find **ReviewRanker** and click the pin icon → it appears in your toolbar

**Step 6 — Test it**
1. Make sure `python app/app.py` is running
2. Open any Amazon product page with customer reviews
3. Wait ~2 seconds — colored score badges appear automatically on each review
4. Reviews are reordered so the most helpful one appears first

**Reloading after code changes:**
Go to `chrome://extensions` → find ReviewRanker → click the **↺ refresh icon**

---------------------------------------------------------------------------------------------------------------

## 10. Deployment (Free, No Fee)

### Option A — Render (Recommended, 100% Free)

Render offers free web service hosting with automatic GitHub deployments.

**Step 1 — Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/SaraDawood2004/rr_ai.git
git push -u origin main
```

**Step 2 — Deploy on Render**
1. Go to https://render.com → sign up free
2. Click **"New"** → **"Web Service"**
3. Connect your GitHub repo
4. Fill in these settings:

| Field          | Value            |
|----------------|-------------------
| Runtime       | Python 3 |
| Build Command | `./render_build.sh` |
| Start Command | `gunicorn app.app:app` |
| Instance Type | **Free** |

5. Click **"Create Web Service"** → Render builds and deploys automatically
6. Your app is live at: `https://your-app-name.onrender.com`

> **Note:** On the free tier, the server sleeps after 15 minutes of inactivity and takes ~30 seconds to wake up on the first request. This is expected.

**Step 3 — Update the plugin to use your live URL**
In `plugin/content.js` and `plugin/popup.js`, change:
```javascript
const API = 'http://localhost:5000'
// → change to:
const API = 'https://your-app-name.onrender.com'
```

### Option B — Railway (Also Free)
1. Go to https://railway.app → sign in with GitHub
2. Click **"New Project"** → **"Deploy from GitHub repo"**
3. Set start command: `gunicorn app.app:app`
4. Free tier gives 500 hours/month

### Option C — Hugging Face Spaces (Free)
Suitable if you want to demo it as a web app only (no plugin).
1. Go to https://huggingface.co/spaces
2. Create a new Space → select **Gradio** or **Flask**
3. Upload your repo


My choice - 

---------------------------------------------------------------------------------------------------------------

## 11. Dataset

**Amazon Fine Food Reviews**
- Source: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
- Originally from Stanford SNAP: https://snap.stanford.edu/data/web-FineFoods.html
- Size: 568,454 reviews spanning Oct 1999 – Oct 2012
- Key columns used: `Text`, `Summary`, `Score`, `HelpfulnessNumerator`, `HelpfulnessDenominator`, `UserId`, `ProductId`
- After filtering (min 5 votes) + sampling: **10,000 stratified reviews**

**Citation:**
> J. McAuley and J. Leskovec. *From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews.* WWW, 2013.

---------------------------------------------------------------------------------------------------------------

## 12. Model Performance

| Model              | RMSE     |   MAE    |    R²  |
|-------             |----------|----------|--------|
| Random Forest ✅  |   0.2552   | 0.2088 | 0.360 |
| XGBoost            |  0.2562    | 0.2081 | 0.355 |

**Top 5 most important features:**
1. `score` (star rating) — 44.0%
2. `char_count` — 6.3%
3. `product_review_count` — 2.9%
4. `adv_ratio` — 2.7%
5. `vader_compound` — 2.7%

--------------------------------------------------------------------------------------------------------------

## 13. Technologies Used

| Layer         |               Technology                 |
|---------------|-------------------------------------------|
| Language      | Python 3.9+ |
| NLP           | spaCy, VADER Sentiment, textstat |
| ML            | scikit-learn (Random Forest), XGBoost |
| Data          | Pandas, NumPy |
| API           | Flask, Flask-CORS |
| Deployment    | Gunicorn, Render |
| Plugin        | Chrome Extension (Manifest V3), Vanilla JS |
| Visualization | Matplotlib, Seaborn |

---------------------------------------------------------------------------------------------------------------

*Built as an academic NLP project. Dataset credits to Stanford SNAP and Amazon.*
~ Sara Dawood S