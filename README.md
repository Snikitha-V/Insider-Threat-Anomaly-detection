# Insider Threat Anomaly Detection Project

## Project Overview
This project applies unsupervised and supervised machine learning techniques to detect insider threats using behavioral, activity, and psychometric data from a realistic enterprise dataset.

### Key Features
- Uses multiple data sources: logon, device, email, file, web, and psychometric data.
- Focuses on anomaly detection to identify users with suspicious or abnormal behavior.
- Evaluates models using ground truth insider labels.

## Workflow Summary
1. **Data Preparation**
   - Load and preprocess all relevant CSV files.
   - Normalize user IDs and merge features.
2. **Exploratory Data Analysis (EDA)**
   - Visualize distributions, correlations, and behavioral patterns.
3. **Modeling (Phase 4)**
   - Train unsupervised models (Isolation Forest, LOF, One-Class SVM, Autoencoder).
   - Save trained models in the `models/` directory.
   - Generate anomaly scores and binary labels for each user.
   - Save results to `results/anomaly_scores_by_user.csv`.
4. **Evaluation (Phase 5)**
   - Merge predictions with ground truth labels from `answers_r42.csv`.
   - Calculate accuracy, precision, recall, and F1-score for anomaly detection.
   - Visualize confusion matrix, ROC, and PR curves.
   - Save evaluation reports in `results_phase5/`.

## Results & Metrics
- **Main unsupervised model used:** Isolation Forest
- **Results file:** `results/anomaly_scores_by_user.csv` contains anomaly scores and labels for each user.
- **Evaluation:**
  - Accuracy and classification metrics are computed by comparing predicted anomaly labels to ground truth insider labels.
  - Results are printed in the notebook and saved to `results_phase5/unsupervised_model_scores.txt`.
- **How accuracy is calculated:**
  1. The Isolation Forest model predicts anomaly labels for each user.
  2. These predictions are merged with ground truth labels.
  3. Standard metrics (accuracy, precision, recall, F1-score) are computed and saved.

## How to Reproduce Results
- Run the notebooks in order:
  1. `phase2_eda.ipynb` for EDA
  2. `phase3_anamoly.ipynb` and `phase4_modeling.ipynb` for model training and scoring
  3. `phase5_evaluation.ipynb` and `phase5_retry.ipynb` for evaluation and reporting
- All outputs and reports are saved in the `results/` and `results_phase5/` folders.

## Presentation Highlights
- **End-to-end workflow:** From raw data to actionable anomaly detection and evaluation
- **Robust evaluation:** Multiple metrics and visualizations for model performance
- **Class imbalance handling:** SMOTE and threshold tuning for realistic detection
- **Interpretability:** Feature importance analysis using Random Forest

## Notes
- If you wish to evaluate other models (LOF, One-Class SVM, Autoencoder), generate and save their predictions in separate columns or files.
- The project is designed for extensibility and can be adapted to other insider threat datasets.

## File Structure
- `models/` — Saved model files
- `results/` — Anomaly scores and labels
- `results_phase5/` — Evaluation reports and plots
- `Notebook/` — Jupyter notebooks for each phase

---
For more details, see the notebooks and scripts in the project root and `Notebook/` folder.