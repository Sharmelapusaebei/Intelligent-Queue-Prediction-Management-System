# ⏱️ Intelligent Queue Prediction & Management System

An end-to-end machine learning solution designed to predict waiting times, analyze peak traffic periods, and optimize staff allocation for campus eateries and administrative service points.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Domain & Operational Context](#-domain--operational-context)
3. [System Architecture & ML Pipeline](#-system-architecture--ml-pipeline)
4. [Project Structure](#-project-structure)
5. [Installation & Setup](#-installation--setup)
6. [Running the Dashboard](#-running-the-dashboard)
7. [Model Evaluation & XAI](#-model-evaluation--xai)
8. [License](#-license)

---

## 🚀 Project Overview
University students and campus staff frequently experience long, unpredictable queues at cafeterias and service counters during peak hours. This project implements a complete machine learning lifecycle—from historical data acquisition and feature engineering to model optimization, explainability (SHAP), and deployment via an interactive Streamlit dashboard.

---

## 🕒 Domain & Operational Context
Unlike generic queuing models, this system is tuned to real-world campus operational constraints:
* **Operating Hours:** 9:00 AM to 5:00 PM (Monday–Friday).
* **Primary Student Rush:** 11:30 AM to 2:30 PM peak window.
* **Admissions Staff Break Wave:** A distinct secondary surge at **~2:00 PM (14:00)** as administrative and admissions block personnel step out for lunch.

---

## 🧠 System Architecture & ML Pipeline
* **Data Preprocessing & Encoding:** `scikit-learn` Pipeline utilizing `StandardScaler` for numerical features (`current_queue_length`, `hour_of_day`) and custom temporal flags.
* **Models Evaluated:** Linear Regression (Baseline), Random Forest Regressor, and XGBoost Regressor (Optimized).
* **Evaluation Metrics:** Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and $R^2$ Score.
* **Explainability:** Feature importance analysis highlighting the impact of rush hours and staff break surges.

---

## 📂 Project Structure
```text
intelligent-queue-system/
│
├── data/                      # Raw and processed CSV logs
├── models/                    # Saved pipeline & model artifacts (.pkl)
├── notebooks/                 # Jupyter notebooks for EDA & experimentation
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Data cleaning & feature engineering
│   ├── train.py               # Model training & hyperparameter tuning
│   └── evaluate.py            # Metrics & SHAP explainability
│
├── app.py                     # Streamlit interactive dashboard
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
