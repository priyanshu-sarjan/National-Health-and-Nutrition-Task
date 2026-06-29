# NHANES Health-Based Age Prediction Hackathon

This repository contains a machine learning solution for the age prediction challenge organized by the **Consulting and Analytics Club, IIT Guwahati** for **Summer Analytics 2026**. 

The goal of this project is to analyze clinical, behavioral, and demographic features from the National Health and Nutrition Examination Survey (NHANES) to predict whether an individual is a **Senior (65+)** or an **Adult/Other**.

## 📌 Project Overview
- **Objective:** Binary classification to predict the `age_group` target variable.
- **Dataset:** 6,287 rows and 7 core healthcare metrics (BMI, Fasting Glucose, Insulin, etc.).
- **Evaluation Metric:** Area Under the ROC Curve (ROC-AUC) / Accuracy.

## 🗂️ Project Directory Structure
```text
├── data/                    # Contains train and test datasets
├── notebooks/               # Jupyter notebooks for interactive EDA & modeling
├── src/                     # Production-ready execution script
├── submissions/             # Generated submission deliverable files
├── .gitignore               # Files excluded from version control
├── README.md                # Comprehensive project report
└── requirements.txt         # Project dependencies
