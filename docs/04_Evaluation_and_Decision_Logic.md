# ClaimShield AI: Evaluation & Decision Logic

This document covers how we mathematically evaluate the models in `src/models/evaluate.py` and how the resulting predictions are translated into actionable business decisions in `src/risk/decision.py`.

---

## 1. Navigating Imbalanced Data

Insurance fraud is highly imbalanced. Typically, 95% of claims are legitimate, and only 5% are fraudulent. 

**The Accuracy Trap:** 
If a model predicts *every* claim is legitimate, it will be 95% accurate. However, it will have caught zero frauds, rendering it useless for the business. Therefore, we **do not** use Accuracy to evaluate our models.

We focus entirely on metrics that isolate the minority class (Fraud):

### A. Recall (Sensitivity)
*   **Formula:** `True Positives / (True Positives + False Negatives)`
*   **What it means:** Out of all the claims that were *actually* fraudulent in reality, what percentage did our model successfully catch?
*   **Business Impact:** Maximizing Recall minimizes financial loss to the company by ensuring fraudsters don't slip through the net.

### B. Precision
*   **Formula:** `True Positives / (True Positives + False Positives)`
*   **What it means:** Out of all the claims our model *flagged* as fraud, what percentage were *actually* fraud?
*   **Business Impact:** Maximizing Precision minimizes operational costs. If Precision is too low, the Special Investigations Unit (SIU) wastes time reviewing innocent claims, and legitimate customers suffer delayed payouts.

### C. The F1-Score
*   **Formula:** `2 * (Precision * Recall) / (Precision + Recall)`
*   **What it means:** The harmonic mean of Precision and Recall. It provides a single metric that balances the trade-off between catching everyone (Recall) and not crying wolf (Precision).

---

## 2. Model Selection: PR-AUC vs. ROC-AUC

In `views/model_performance.py`, we display both ROC and Precision-Recall (PR) curves.

*   **ROC-AUC (Receiver Operating Characteristic):** Plots True Positive Rate against False Positive Rate. It is the industry standard for balanced datasets, but it can be overly optimistic when dealing with imbalanced data.
*   **PR-AUC (Precision-Recall Area Under Curve):** Plots Precision against Recall. **This is our primary metric.** PR-AUC is highly sensitive to the minority class. Our champion model (XGBoost) was selected specifically because it maximized the PR-AUC, proving it could identify true fraud without raising unacceptable levels of false alarms.

---

## 3. The Decision Engine (`src/risk/decision.py`)

A raw probability score is not a business action. We map the calibrated probability (0-100) to distinct operational tiers:

1.  **LOW RISK (0 - 30): `APPROVE`**
    *   *Action:* Straight-Through Processing (STP). The claim is automatically approved and paid out.
    *   *Value:* Maximizes customer satisfaction and drastically reduces manual operational costs.
2.  **MEDIUM RISK (31 - 70): `REVIEW`**
    *   *Action:* Routed to a standard claims adjuster for a manual sanity check.
3.  **HIGH RISK (71 - 100): `INVESTIGATE`**
    *   *Action:* Hard stop. The claim is escalated directly to the SIU (Special Investigations Unit) for a deep dive, supported by the SHAP explanation output.

---

## 4. Financial Exposure Prioritization

The SIU team cannot investigate 100 claims simultaneously. They need a prioritization metric.

We calculate **Estimated Financial Exposure** as:
`Exposure = Claim_Amount * Fraud_Probability`

**Example:**
*   Claim A: $10,000 claim with a 90% risk score. (Exposure = $9,000)
*   Claim B: $100,000 claim with a 40% risk score. (Exposure = $40,000)

Even though Claim A is far more likely to be fraud, Claim B poses a significantly higher financial threat to the enterprise. The Dashboard (`views/claims.py`) allows investigators to sort their queue by Exposure to protect the bottom line efficiently.

---

## 5. The KNN Similarity Engine

In `views/investigation.py` (Step 5), the system surfaces "Historical Context (Similar Claims)".

*   **The Math:** We use the **K-Nearest Neighbors (KNN)** algorithm with **Cosine Similarity**.
*   **The Process:** Every claim is transformed into a multi-dimensional mathematical vector during preprocessing. When a new claim is investigated, the KNN engine calculates the cosine angle between the new claim's vector and every historical claim's vector in the database.
*   **The Output:** It returns the top 3 claims with the smallest angles (highest similarity percentages). 
*   **Business Value:** This allows investigators to rely on precedent. If a new claim is 94% similar to a historical claim that was definitively proven to be fraudulent, the investigator has a massive head start on where to look.
