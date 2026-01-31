# StreamPulse – Module 1: ML Workflow Setup & Baseline Churn Prediction

## 📌 Project Overview

This project represents **Module 1** of the *StreamPulse* machine learning pipeline. The goal is to establish a complete, reproducible **end-to-end ML workflow** using a baseline supervised learning model to predict **customer churn**.

The module covers:

* Data loading and inspection
* Feature preprocessing (scaling + encoding)
* Model training and evaluation
* Cross-validation and hyperparameter tuning
* Model persistence for future use

This baseline serves as a reference point for more advanced modeling in later modules.

---

## 🛠️ Technologies Used

* **Python 3**
* **Pandas / NumPy** – data handling
* **Scikit-learn** – preprocessing, modeling, evaluation
* **Joblib** – model persistence

---

## 📂 Project Structure

```text
stream_pulse_project/
├── data/
│   └── Stream_pulse_customer_data.csv
├── models/
│   └── baseline_model.pkl
├── src/
│   ├── m1_baseline_model.py
│   └── preprocessing_pipeline.pkl
└── README.md
```

---

## 📊 Dataset Description

The dataset contains customer-level information related to usage behavior and subscription details.

### Key Columns:

* `customer_id` – unique customer identifier
* `age` – customer age
* `country` – customer country
* `subscription_type` – Free / Basic / Premium
* `monthly_spend` – average monthly spending
* `sessions_per_month` – usage frequency
* `avg_watch_time_min` – average watch time
* `churned` – **target variable** (1 = churned, 0 = not churned)

---

## ⚙️ ML Workflow Summary

### 1. Data Preprocessing

* Categorical features encoded using **OneHotEncoder**
* Numerical features standardized using **StandardScaler**
* Preprocessing handled via **ColumnTransformer**

### 2. Train-Test Split

* 70% training / 30% testing
* **Stratified split** to preserve churn distribution

### 3. Baseline Model

* **Logistic Regression** classifier
* Implemented using a unified Scikit-learn **Pipeline**

### 4. Model Evaluation

Metrics computed on the test set:

* Accuracy
* Precision
* Recall

### 5. Cross-Validation

* 5-fold cross-validation on training data
* Used to assess model stability

### 6. Hyperparameter Tuning

* GridSearchCV with 3 parameter combinations
* Optimized regularization strength (`C`)

---

## 📈 Results Interpretation

* The baseline model achieves ~81% accuracy
* Precision and Recall for churn are low due to **class imbalance**
* This behavior is expected for a baseline churn model

These results motivate more advanced techniques in future modules, such as:

* Class weighting
* Resampling (SMOTE)
* Tree-based models

---

## 💾 Saved Artifacts

After execution, the following files are generated:

* `models/baseline_model.pkl` – trained Logistic Regression model
* `src/preprocessing_pipeline.pkl` – preprocessing pipeline

These artifacts can be reused directly for inference or further training.

---

## ▶️ How to Run

From the project root directory:

```bash
python src/m1_baseline_model.py
```

---

## ✅ Module Status

* ✔ End-to-end ML workflow implemented
* ✔ Baseline model trained and evaluated
* ✔ Ready for **Module 2: Feature Engineering & Advanced Models**

---

## 📌 Notes

This module focuses on **correctness and structure**, not optimal churn detection performance. Improving recall and precision is addressed in later stages of the project.

---

**Author:** StreamPulse Project
**Module:** ML Workflow Setup & Baseline Churn Prediction
