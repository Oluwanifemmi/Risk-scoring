# 🏦 Credit Risk Scoring API

A production ready machine learning API that predicts the probability of loan default, classifies applicants into risk tiers, and (optionally) explains the decision in plain English using an LLM. Built end-to-end: EDA → feature engineering → imbalanced classification → model serving → containerization.

---

## What this does

Given an applicant's financial and demographic profile, the API returns:

- **Probability of default** : a calibrated score between 0 and 1
- **Binary prediction** : will they default or not
- **Risk tier** :`Low Risk` / `Medium Risk` / `High Risk`

---

## Model performance

Trained on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset (~100K+ applicants, 100+ raw features, heavily imbalanced target).

| Metric | Score |
|---|---|
| Mean AUC (5-fold CV) | **0.7345** |
| Mean Gini | **0.4691** |
| KS Statistic | **0.3514** (p < 0.0001) |

These numbers reflect a real, imbalanced credit dataset: Gini ~0.47 and KS ~0.35 are solidly in the range banks consider usable for risk segmentation, not just a toy demo.

---

## Under the hood

The model isn't a single classifier, it's a full `imblearn` pipeline that handles missing data, encoding, outliers, class imbalance, and scaling before XGBoost ever sees a row:

```
ColumnTransformer
 ├── Mean imputation        → continuous features (EXT_SOURCE_*, AMT_*, etc.)
 ├── Most-frequent imputation + frequency encoding → sparse categoricals
 ├── Frequency encoding      → high-cardinality categoricals
 └── One-hot encoding        → low-cardinality categoricals
        ↓
NamedWinsorizer (custom transformer)  → caps outliers at the 5th/95th percentile
        ↓
SMOTE                                  → rebalances the minority (default) class
        ↓
StandardScaler
        ↓
XGBClassifier (scale_pos_weight tuned for class imbalance)
```

**Custom transformer:** `NamedWinsorizer` in `custom_transformers.py` is a scikit-learn compatible transformer that preserves column names through the pipeline while capping outliers necessary because most winsorization implementations return bare numpy arrays and break downstream `ColumnTransformer` steps.

**Explainability:** SHAP is used offline (see the notebook) to validate feature level contributions per prediction, ensuring the model's decisions are auditable, a hard requirement for any real credit risk system.

---

## Tech stack

| Layer | Tool |
|---|---|
| API framework | FastAPI |
| Server | Uvicorn |
| ML pipeline | scikit-learn, imbalanced-learn, feature-engine |
| Model | XGBoost |
| Explainability | SHAP |
| LLM explanations | OpenAI-compatible chat completion |
| Serialization | joblib |
| Containerization | Docker |

---

## Project structure

```
.
├── main.py                    # FastAPI app + prediction endpoints
├── custom_transformers.py     # NamedWinsorizer (required to unpickle the model)
├── creditriskscore.pkl        # Trained pipeline (preprocessing + XGBoost)
├── new-credit.ipynb           # Full EDA → training → evaluation notebook
├── requirements.txt
└── Dockerfile
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Oluwanifemmi/Risk-scoring.git
cd Risk-scoring
```

### 2. Build the Docker image
```bash
docker build -t risk-app .
```

### 3. Run the container
```bash
docker run -p 8000:8000 risk-app
```
> If your `main.py` uses the LLM explanation feature, pass your API key instead:
> ```bash
> docker run -p 8000:8000 -e OPENAI_API_KEY=your-key-here risk-app
> ```

### 4. Open the aws link
Visit **[[http://localhost:8000/docs](http://13.40.7.247:8000/docs)]** 

---

## API

### `POST /predict_probability`

Returns the raw model output: probability, binary prediction, and risk tier.

**Request body:**
```json
{
  "CODE_GENDER": "M",
  "AMT_CREDIT": 450000,
  "AMT_ANNUITY": 27000,
  "AMT_GOODS_PRICE": 400000,
  "NAME_INCOME_TYPE": "Working",
  "NAME_EDUCATION_TYPE": "Higher education",
  "NAME_FAMILY_STATUS": "Married",
  "REGION_POPULATION_RELATIVE": 0.018,
  "DAYS_BIRTH": -14600,
  "DAYS_EMPLOYED": -2100,
  "DAYS_REGISTRATION": -3500,
  "DAYS_ID_PUBLISH": -1800,
  "OWN_CAR_AGE": 8,
  "FLAG_EMP_PHONE": 1,
  "REGION_RATING_CLIENT": 2,
  "REGION_RATING_CLIENT_W_CITY": 2,
  "REG_CITY_NOT_LIVE_CITY": 0,
  "REG_CITY_NOT_WORK_CITY": 1,
  "ORGANIZATION_TYPE": "Business Entity Type 3",
  "EXT_SOURCE_1": 0.55,
  "EXT_SOURCE_2": 0.62,
  "EXT_SOURCE_3": 0.48,
  "OCCUPATION_TYPE": "Laborers",
  "APARTMENTS_AVG": 0.1237,
  "ELEVATORS_AVG": 0.0345,
  "FLOORSMAX_AVG": 0.1667,
  "FLOORSMIN_AVG": 0.0833,
  "LIVINGAREA_AVG": 0.1502,
  "APARTMENTS_MODE": 0.125,
  "FLOORSMAX_MODE": 0.1667,
  "FLOORSMIN_MODE": 0.0833,
  "LIVINGAREA_MODE": 0.1489,
  "APARTMENTS_MEDI": 0.1240,
  "FLOORSMAX_MEDI": 0.1667,
  "FLOORSMIN_MEDI": 0.0833,
  "LIVINGAREA_MEDI": 0.1495,
  "TOTALAREA_MODE": 0.1310,
  "DAYS_LAST_PHONE_CHANGE": -500,
  "FLAG_DOCUMENT_3": 1
}
```

**Response:**
```json
{
  "probability_default": 0.4842,
  "predicted_default": 0,
  "risk_tier": "Medium Risk"
}
---
