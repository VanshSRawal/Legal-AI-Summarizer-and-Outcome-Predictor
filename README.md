# ⚖️ Legal AI: FIRAC Summarizer & Case Outcome Predictor (IT Act, 2000)

## Overview
This project is a domain-specific Legal AI system focused on Indian cyber law cases under the Information Technology Act, 2000.
It consists of two tightly integrated phases:

1. Phase 1 – FIRAC Summarizer (RAG-based)
2. Phase 2 – Case Outcome Predictor (Explainable ML)

The system is designed to assist legal research, not replace judicial reasoning.

---

## Phase 1: FIRAC Summarizer (RAG Pipeline)

**What it does**
- Generates FIRAC-style summaries (Facts, Issues, Rules, Analysis, Conclusion)
- Grounds outputs in the IT Act, 2000 and uploaded judgments

**How it works**
- PDF parsing and text extraction
- Semantic chunking
- Vector similarity search
- Context-aware summary generation

**Why RAG**
- Reduces hallucination
- Ensures statute-grounded summaries
- Improves explainability

---

## Phase 2: Case Outcome Predictor

**What it does**
- Predicts Petitioner vs Respondent outcome
- Outputs probability, threshold, explanation, and penalties

**Model**
- TF-IDF (uni + bi-grams)
- IT Act section one-hot encoding
- Calibrated Logistic Regression (Platt scaling)

**Why Logistic Regression**
- Interpretable
- Stable on legal datasets
- Supports explainable AI

---

## Evaluation

**Quantitative**
- Accuracy
- Macro F1-score
- ROC-AUC

**Qualitative**
- Manual comparison with real judgments
- Validation of sections, outcomes, and explanations

---

## Explainability
- Feature contribution analysis
- Key words and sections
- Plain-English explanation for non-technical users

---

## Penalty Mapping
- Section-aware IT Act penalty retrieval
- Educational and research use only

---

## User Interface
- Streamlit-based web application
- Unified summarizer + predictor interface
- PDF upload and result visualization

---

## Project Structure
```
app_integrated.py
ik_it_act_scraper.py
clean_preprocess_it_cases.py
train_models2.py
case_predictor.py
explainer.py
narrative_explainer.py
penalties_retriever.py
it_act_config.py
models/
scraped_it_cases/
README.md
```

---

## Workflow Diagram
![Workflow Diagram Placeholder](<img width="865" height="342" alt="image" src="https://github.com/user-attachments/assets/4d01a6de-eb98-435b-a711-19a9a3a78434" />
)

---

## Future Work
- Non-IT Act detection guardrails
- SHAP-based explanations
- External legal expert validation
- API-based deployment

---

## Disclaimer
This project is for academic and research purposes only.
It does not provide legal advice.

---

## References
- Information Technology Act, 2000
- Indian Kanoon
- Scikit-learn Documentation
