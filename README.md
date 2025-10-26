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
3. **Modeling (Phase 3 & 4 - Unsupervised)**
   - Train unsupervised models (Isolation Forest, LOF, One-Class SVM, Autoencoder).
   - Save trained models in the `models/` directory.
   - Generate anomaly scores and binary labels for each user.
   - Save results to `results/anomaly_scores_by_user.csv`.
4. **Evaluation & Supervised (Phase 5)**
   - Merge predictions with ground truth labels from `answers_r42.csv`.
   - Train a supervised Random Forest on features (with SMOTE for imbalance) and tune a decision threshold on a validation split.
   - Calculate accuracy, precision, recall, and F1-score for both unsupervised predictions and supervised RF probabilities.
   - Visualize confusion matrix, ROC, and PR curves.
   - Save evaluation reports in `results_phase5/`.

## Results & Metrics

### Unsupervised (Isolation Forest)
- Source file: `results/anomaly_scores_by_user.csv` (per-user scores and labels)
- Evaluation summary (from `results_phase5/unsupervised_model_scores.txt`):
   - Accuracy: 0.904
   - Insider precision: 0.24, recall: 0.17, F1: 0.20 (class imbalance makes insider detection hard without threshold tuning)
   - Normal precision: 0.94, recall: 0.96, F1: 0.95

How metrics are computed: predicted anomaly labels are compared to ground truth insider labels after aligning users.

### Supervised Approach Results (Random Forest + SMOTE)
The supervised model is trained on the engineered user feature set with SMOTE applied to the training split and a tuned probability threshold. Key evaluations on the full merged dataset:

- From `results_phase5/evaluation_report_full.txt` (Best threshold: 0.14):

   Confusion Matrix
   ```
   [[924   6]
    [ 11  59]]
   ```

   Classification Report
   ```
                        precision    recall  f1-score   support

            Normal     0.99      0.99      0.99       930
          Insider      0.91      0.84      0.87        70

         accuracy                         
       macro avg      0.95      0.92      0.93      1000
   weighted avg       0.98      0.98      0.98      1000
   ```

- From `results_phase5/evaluation_report_full_merged.txt` (Best threshold: 0.40):

   Confusion Matrix
   ```
   [[925   5]
    [ 13  57]]
   ```

   Classification Report
   ```
                        precision    recall  f1-score   support

            Normal       0.99      0.99      0.99       930
          Insider       0.92      0.81      0.86        70

         accuracy                           0.98      1000
       macro avg       0.95      0.90      0.93      1000
   weighted avg       0.98      0.98      0.98      1000
   ```
Note: A smaller diagnostic split is also saved in `results_phase5/evaluation_report.txt`, but the full-dataset evaluations above are the primary supervised results for presentation.

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
