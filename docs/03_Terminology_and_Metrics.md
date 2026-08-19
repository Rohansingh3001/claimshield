# ClaimShield AI: Terminology & Metrics Cheatsheet

When defending the project, the precise terminology you use matters heavily. Use this guide to answer technical metric questions.

## 1. Dealing with Imbalanced Data
In fraud detection, datasets are inherently imbalanced (e.g., 95% legitimate, 5% fraud). 
* **The Trap:** If an evaluator asks, *"Why didn't you just use Accuracy to evaluate the model?"*
* **Your Answer:** *"Accuracy is a deceptive metric for fraud. If I build a dumb model that just says 'Not Fraud' to every single claim, it will be 95% accurate, but it will fail to catch a single fraudulent claim. That's why we focused on Recall, Precision, and PR-AUC."*

## 2. Evaluation Metrics

### Recall (The most important metric for Fraud)
* **Definition:** Out of all the *actual* frauds that occurred, what percentage did our model successfully catch?
* **Business Impact:** High recall minimizes financial loss because fewer frauds slip through the cracks.

### Precision 
* **Definition:** Out of all the claims our model *flagged* as fraud, what percentage were *actually* fraud?
* **Business Impact:** High precision minimizes false alarms. If precision is too low, investigators waste time reviewing legitimate claims, and customers get annoyed by delayed payouts.

### F1-Score
* **Definition:** The harmonic mean (balance) between Precision and Recall. Used when you want to compromise between catching all frauds and not having too many false alarms.

### ROC-AUC vs. PR-AUC
* **ROC-AUC (Receiver Operating Characteristic):** Measures the general ability of the model to distinguish between classes. Good for balanced data.
* **PR-AUC (Precision-Recall AUC):** Measures the tradeoff between Precision and Recall. 
* **If asked:** *"We prioritized PR-AUC over ROC-AUC because our dataset is imbalanced. PR-AUC is specifically sensitive to the minority class (fraud), making it the gold standard for this use case."*

## 3. The Decision Engine (Risk Thresholds)
We don't just output a probability; we map it to business actions using a tiered threshold system (`src/risk/decision.py`):
1. **0 - 30 (APPROVE):** Fast-tracked straight-through processing. High customer satisfaction.
2. **31 - 70 (REVIEW):** Requires a light manual review by a junior adjuster.
3. **71 - 100 (INVESTIGATE):** Escalated immediately to the Special Investigations Unit (SIU).

## 4. Financial Exposure Estimate
Calculated as: `Claim_Amount` * `Fraud_Probability`.
* **Why it matters:** If the SIU team has 50 claims in their queue, they need to know which one to investigate first. By multiplying the risk percentage by the dollar amount, we allow them to prioritize claims that pose the highest financial threat to the company.
