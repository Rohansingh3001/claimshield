# ClaimShield AI: Machine Learning & Algorithms

## 1. The Core Algorithms
We experimented with three distinct algorithms. If the evaluators ask why we chose these three, answer: *"We started with a simple linear baseline (Logistic Regression), moved to a standard ensemble (Random Forest), and finalized with a high-performance boosting algorithm (XGBoost) to maximize our recall on fraudulent claims."*

### A. Logistic Regression (The Baseline)
A simple statistical model that tries to draw a straight line (linear boundary) between Fraud and Non-Fraud. 
* **Pros:** Fast, simple, highly explainable.
* **Cons:** Cannot capture complex, non-linear relationships in data.

### B. Random Forest (The Standard Ensemble)
An ensemble method that builds hundreds of different "Decision Trees" and lets them vote on whether a claim is fraud.
* **Pros:** Reduces overfitting, handles non-linear data well.
* **Cons:** Slower, and often struggles with highly imbalanced data compared to boosting methods.

### C. XGBoost (The Champion Model)
XGBoost stands for **Extreme Gradient Boosting**. Unlike Random Forest where trees are built independently, XGBoost builds trees *sequentially*. Each new tree specifically tries to correct the errors made by the previous tree.
* **Why we chose it:** It natively handles missing data, trains extremely fast, and provides state-of-the-art accuracy on tabular (CSV) data. It gave us the best **PR-AUC** (Precision-Recall Area Under Curve).

---

## 2. Model Calibration (CalibratedClassifierCV)
Machine learning models naturally output a raw score, not a true probability. If XGBoost outputs `0.8`, it doesn't strictly mean there is an 80% chance of fraud. 
* **How we fixed it:** We wrapped our models in Scikit-Learn's `CalibratedClassifierCV` (using the Sigmoid method, also known as Platt Scaling). This mathematically maps the model's raw output into a true 0-100 **Fraud Risk Score**. 
* **If asked:** *"We used probability calibration so that when our dashboard tells an investigator the risk score is 85/100, it reflects true statistical confidence, rather than just an arbitrary model threshold."*

---

## 3. Explainable AI (SHAP)
Predicting fraud isn't enough; the business needs to know *why*. We implemented **SHAP (SHapley Additive exPlanations)**.
* **What it is:** A concept from Game Theory that calculates the exact contribution of each individual feature to the final prediction.
* **How we use it:** Instead of a "black box" telling the investigator a claim is fraud, SHAP provides a visual breakdown: *"Risk is high BECAUSE the claim amount is unusually large AND the policy is only 2 months old."*

---

## 4. The Similarity Engine (KNN)
We implemented a feature to find "Historical Similar Claims". 
* **Algorithm used:** **K-Nearest Neighbors (KNN)** using **Cosine Similarity**.
* **How it works:** When a new claim comes in, the engine converts it into a vector (a point in space) and measures the distance (Cosine angle) to all historical claims. 
* **Business Value:** It shows investigators past cases that looked exactly like the current one, providing precedent on how those past cases were resolved.
