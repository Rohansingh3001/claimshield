# ClaimShield AI: Machine Learning & Explainable AI (XAI)

This document covers the core machine learning logic implemented in `src/models/`, exploring the algorithms chosen, how they are calibrated to reflect reality, and how we achieve explainability to ensure regulatory compliance and investigator trust.

---

## 1. Algorithm Selection (`src/models/train.py`)

We do not rely on a single model. The pipeline dynamically trains and evaluates three distinct algorithms to establish baselines and select a champion.

### A. Logistic Regression (The Linear Baseline)
*   **Theory:** Assumes a linear relationship between features and the probability of fraud. It draws a hyperplane separating the classes.
*   **Role:** Acts as our sanity check. If advanced models cannot significantly outperform Logistic Regression, the data lacks complex non-linear patterns, and we should stick to the simpler model for maximum explainability.

### B. Random Forest (Bagging Ensemble)
*   **Theory:** Builds hundreds of independent Decision Trees in parallel (bagging). Each tree gets a random subset of data and features. They vote on the final outcome.
*   **Role:** Excellent at handling non-linear relationships without heavily overfitting.

### C. XGBoost (Boosting Ensemble - The Champion)
*   **Theory:** Extreme Gradient Boosting builds trees *sequentially*. Tree #1 makes predictions. Tree #2 looks specifically at where Tree #1 was wrong, and tries to fix those specific errors. Tree #3 fixes Tree #2's errors, and so on.
*   **Role:** This is our production model. It inherently handles missing data, trains rapidly via hardware optimization, and consistently provides state-of-the-art results on tabular data (which is what insurance claims are).

---

## 2. Probability Calibration (The Math Behind the Score)

**Crucial Concept:** Machine learning classifiers do not output true probabilities. They output a raw score that places a data point on one side of a decision boundary. 

If XGBoost outputs `0.85`, it **does not** mean there is an 85% mathematical probability of fraud.

Because ClaimShield AI acts as a decision support system, investigators rely on the **Fraud Risk Score (0-100)** to be statistically accurate. If we tell them a claim is 90% risky, 9 out of 10 claims with that score must actually be fraud.

**The Solution (`CalibratedClassifierCV`):**
In `src/models/train.py`, after training XGBoost, we wrap the model in Scikit-Learn's `CalibratedClassifierCV`.
*   We use the **Sigmoid** method (Platt Scaling).
*   This fits a secondary logistic regression model on top of the raw outputs of the XGBoost model, mathematically squashing the raw scores into a true, calibrated probability distribution between 0.0 and 1.0. 
*   We multiply this by 100 to display the Risk Score in the UI.

---

## 3. Explainable AI (SHAP)

Modern AI regulations require that automated decisions affecting consumers be explainable. A "black box" XGBoost model that simply outputs "Fraud" is unacceptable in the insurance industry.

We use **SHAP (SHapley Additive exPlanations)** in `src/models/explain.py`.

### The Theory (Game Theory)
Imagine a team of 3 people (Features A, B, and C) working together to generate a profit (The Risk Score). SHAP mathematically calculates exactly how much money (Risk) each specific person (Feature) brought to the table, based on all possible combinations of them working together.

### The Implementation
1.  When an investigator views a claim in `views/investigation.py`, the backend passes the claim's specific data vector to the SHAP `TreeExplainer`.
2.  The Explainer returns an array of SHAP values corresponding to the exact numerical impact each feature had on pushing the score up (towards fraud) or down (towards legitimate).
3.  We map these raw numerical SHAP values to plain-English business logic in the UI:
    *   *Instead of:* `Claim_Amount_SHAP = +1.2`
    *   *The UI shows:* 🔴 **+12 Risk:** The Claim Amount is unusually high compared to the Coverage Amount.
