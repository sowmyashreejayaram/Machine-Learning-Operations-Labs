# 🎭 Real-Time Sentiment Analysis Pipeline

## Overview
MLOps pipeline for sentiment analysis using Apache Airflow, VADER, and TextBlob.

## Features
- ✨ Analyzes 1000+ social media messages
- 🤖 Multi-model sentiment analysis
- 📊 Beautiful visualizations
- 🔄 Automated with Airflow

## Quick Start

### 1. Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon')"
```

### 2. Run Airflow
```bash
export AIRFLOW_HOME=$(pwd)
airflow db init
airflow users create --username admin --password admin \
    --firstname Admin --lastname User --role Admin --email admin@example.com
```

### 3. Start Services
Terminal 1: `airflow webserver --port 8080`
Terminal 2: `airflow scheduler`

### 4. Access
http://localhost:8080 (admin/admin)

## Pipeline Tasks
1. Generate synthetic data (1000+ messages)
2. Preprocess text (clean, normalize)
3. Sentiment analysis (VADER + TextBlob)
4. Create visualizations (charts, word clouds)

## Results
- Sentiment distribution chart
- Positive/negative word clouds
- CSV reports with scores

## Author
Sowmyashree - Northeastern University MLOps Lab Assignment 2
