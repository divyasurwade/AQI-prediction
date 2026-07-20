# Air Quality Index (AQI) Comparative Analysis & Prediction System

A modern, responsive web application for Air Quality Index (AQI) analysis, machine learning model benchmark evaluation, real-time prediction, multi-city leaderboards, and 24-hour LSTM time-series forecasting.

---

## ⚡ Quick Start Instructions for Client

### Method 1: Easy One-Click Setup (Recommended)

#### 🪟 Windows Users:
Simply double-click the **`run.bat`** file in the project folder.
- It will automatically create a Python virtual environment (`.venv`), install all required dependencies from `requirements.txt`, and start the web application.
- Open your web browser and go to: **`http://localhost:5000`**

#### 🍎 macOS / 🐧 Linux Users:
Open Terminal in the project folder and run:
```bash
chmod +x run.sh
./run.sh
```
- Open your web browser and go to: **`http://localhost:5000`**

---

### Method 2: Manual Terminal Setup

1. **Prerequisites**: Ensure Python 3.9 or higher is installed ([python.org](https://www.python.org/downloads/)).
2. **Open Terminal / Command Prompt** in the project folder.
3. **Create & Activate Virtual Environment**:
   - **Windows**:
     ```cmd
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - **macOS / Linux**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Start the Web Application**:
   ```bash
   python app.py
   ```
6. **Open in Browser**: Open `http://localhost:5000` in Google Chrome, Edge, or Safari.

---

## 🌐 Cloud Deployment Options (GitHub, Vercel & Render)

This repository includes pre-built deployment configurations for one-click cloud hosting:

- **Vercel**: Configured via `vercel.json` for serverless Python web hosting.
- **Render**: Configured via `render.yaml` using Gunicorn (`gunicorn --bind 0.0.0.0:$PORT app:app`).

---

## 📂 Project Structure

```
AQI/
├── app.py                # Main Flask Web Application & REST APIs
├── requirements.txt      # Python dependencies
├── vercel.json           # Vercel serverless configuration
├── render.yaml           # Render deployment configuration
├── run.bat               # One-click Windows runner
├── run.sh                # One-click Mac/Linux runner
├── ml/
│   ├── data_generator.py # Synthetic & real air quality data generator
│   ├── preprocessing.py  # Outlier handling & missing value imputation
│   ├── train.py          # Model training pipeline (Linear, ANN, LSTM, Random Forest, SVR)
│   └── forecaster.py     # 24h LSTM time-series forecast engine
├── models/               # Saved trained model checkpoints (.keras, .joblib) & metrics.json
├── static/
│   ├── css/style.css     # Glassmorphism dark mode theme & responsive styles
│   └── js/               # Interactive gauge & Chart.js controllers
└── templates/            # HTML pages (Dashboard, Predictor, Comparison, Forecast, Leaderboard)
```
