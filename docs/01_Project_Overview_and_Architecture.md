# ClaimShield AI: Project Overview & Architecture

## 1. The Business Problem
Insurance companies lose billions of dollars annually to fraudulent claims. However, manually investigating every single claim is too expensive, slow, and negatively impacts the experience for legitimate customers. 

**The Solution:** ClaimShield AI acts as an **Investigator Decision-Support Tool**. It does not automatically reject claims. Instead, it flags suspicious claims, provides a **Risk Score (0-100)**, explains *why* the claim was flagged, and suggests an action (Approve, Review, or Investigate). This optimizes human resources by pointing investigators to the right claims.

## 2. System Architecture & Pipeline Flow
The project is divided into an **ML Pipeline** (backend processing) and a **Streamlit Web Dashboard** (frontend UI). The ML Pipeline (`run_pipeline.py`) follows a strict sequence:

### Step 1: Data Loading (`DataLoader`)
Reads the raw historical claims data (`ClaimShieldAI-Dataset.csv`).

### Step 2: Data Validation (`DataValidator`)
Validates the integrity of the data. A critical part of this step is identifying **Data Leakage**. 
* **What you should say if asked:** "We built a validator to automatically catch data leakage—variables like 'Settlement_Amount' that are known *after* a claim is processed. If we train our model on post-investigation data, the model will cheat and perform artificially well, but fail in the real world."

### Step 3: Feature Engineering (`FeatureEngineer`)
Raw data is often not enough. We create new mathematical columns (features) based on business logic to help the model find patterns. For example, if we have `Claim_Amount` and `Policy_Limit`, we might create a new feature called `Amount_to_Limit_Ratio`.

### Step 4: Data Preprocessing (`DataPreprocessor`)
Machine learning models only understand numbers. This step cleans the data:
* **Missing Values (SimpleImputer):** Fills blank cells. We use the *median* for numbers and a *constant* ('missing') for text.
* **StandardScaler:** Normalizes numerical data so large numbers (like $50,000) don't overshadow small numbers (like Age: 35).
* **OneHotEncoder:** Converts categorical text (e.g., Car Make: "Toyota", "Ford") into binary numbers (0s and 1s).

### Step 5: Model Training (`ModelTrainer`)
We split the data into **Training** (80%) to teach the model and **Testing** (20%) to test it on unseen data. We train multiple algorithms (Logistic Regression, Random Forest, XGBoost) to compare them.

### Step 6: Streamlit UI
Once models are trained and saved (as `.pkl` files using `joblib`), the Streamlit application (`app.py`) loads them to serve predictions interactively to the end user.
