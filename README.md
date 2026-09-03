# Cyberbullying Tweet Classifier

A multiclass NLP classification system that detects and categorizes cyberbullying in English-language tweets. Built as the capstone project for the BNSP Associate Data Scientist (CADS) certification.

🔗 **Live App:** https://bnsppreparation1-3p3xjkqubmmkkf3loxcqqs.streamlit.app/

## Overview

Cyberbullying on social media has grown sharply alongside rising social media use. This project builds a model that classifies tweets into one of six categories — **age, ethnicity, gender, religion, other_cyberbullying, or not_cyberbullying** — to support content moderation and trust & safety workflows.

## Dataset

- Source: [Kaggle — Cyberbullying Classification](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification)
- 47,000+ tweets, balanced across 6 categories (~16.7% each)
- Scoped to English-language tweets only (93.91% of the dataset, confirmed via language detection)

## Pipeline

1. **Data validation** — checked class distribution, spot-checked label consistency, cross-checked dominant terms per category via WordCloud
2. **EDA** — confirmed balanced classes, no missing values, no extreme outliers in tweet length
3. **Text cleaning** — lowercasing, URL/mention/hashtag/punctuation removal, tokenization, stopword removal, POS-aware lemmatization, deduplication
4. **Feature engineering** — TF-IDF vectorization (max 6,000 features, unigram + bigram)
5. **Modeling** — compared 6 algorithms (Logistic Regression, Multinomial NB, Linear SVM, Random Forest, XGBoost, ComplementNB) on identical TF-IDF input
6. **Evaluation** — selected based on F1-macro to weight all classes equally
7. **Deployment** — Streamlit Cloud app with model, vectorizer, and label encoder bundled via joblib

## Results

| Model | F1-macro | Train Time |
|---|---|---|
| **XGBoost** ⭐ | **0.8246** | ~268s |
| Logistic Regression | 0.8125 | ~6.3s |
| Linear SVM | 0.8111 | ~1s |
| Random Forest | Moderate | Moderate |
| Multinomial/ComplementNB | Lower | Fastest |

XGBoost was selected for its highest F1-macro and fast inference time, making it suitable for deployment.

**Confusion matrix insights:**
- `age`, `ethnicity`, `religion` — high precision & recall (>0.94), driven by distinct lexical markers
- Most common confusion: `other_cyberbullying` ↔ `not_cyberbullying`, and `gender` → `not_cyberbullying`
- Model is reliable at distinguishing bullying vs. non-bullying broadly; most errors occur in sub-category assignment

## App Features

- **Disclaimer gate** — required consent before access, since the dataset contains explicit language
- **Language detection** — flags non-English input automatically
- **Low-confidence badge** — predictions under 60% confidence are flagged for manual review
- **WordCloud with double-safety reveal** — gated behind an extra click, resets per prediction
- **EDA & model comparison pages** — full methodology transparency for reviewers/assessors

## Limitations

- English-only; not validated for other languages, including Indonesian
- Labels are pre-labeled by the dataset provider; validation was sampling-based, not a full audit
- Dataset is from 2020 — language and terminology on social media may have shifted since

## Recommendations

- Use as a human-in-the-loop tool: low-confidence predictions should be queued for manual review, not auto-actioned
- Prioritize escalation for high-accuracy categories (age, ethnicity, religion) as fast signals to trust & safety teams
- Retrain periodically to keep up with evolving social media language

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · NLTK · Streamlit · joblib

## Certification

Developed and assessed as part of the **BNSP Associate Data Scientist (CADS)** certification, August 2026.

---
**Author:** M. Fahmi Jamil
