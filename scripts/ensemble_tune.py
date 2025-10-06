"""Ensemble score + CV threshold tuning

Combines available anomaly scores (IsolationForest, LOF, Autoencoder MSE) by
rank-sum and mean-zscore, runs stratified CV on an inner training set to pick
a stable contamination (threshold) and evaluates on a held-out test set.

Saves per-fold tuning CSVs, combined CSV, histogram, JSON summary and
final per-user ensemble scores to Notebook/results/.
"""
import os
import json
import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix

try:
    # optional: load saved autoencoder if tensorflow is available
    from tensorflow.keras.models import load_model
    HAS_TF = True
except Exception:
    HAS_TF = False


def load_inputs(results_path, user_features_path, answers_path):
    # anomaly_scores_by_user.csv is a helpful starting point (may contain iso/lof/ae cols)
    df_scores = None
    p = Path(results_path)
    if p.exists():
        df_scores = pd.read_csv(p)

    # user_features (numeric matrix)
    X_df = pd.read_csv(user_features_path)

    # ground truth answers
    y_df = pd.read_csv(answers_path)

    # Normalize key name for label column if necessary
    if 'label' not in y_df.columns:
        # try second column
        if y_df.shape[1] >= 2:
            y_df = y_df.rename(columns={y_df.columns[1]: 'label'})

    return df_scores, X_df, y_df


def prepare_feature_matrix(X_df):
    # drop common id columns
    drop_candidates = [c for c in ['user_id', 'user', 'employee_id', 'username', 'name'] if c in X_df.columns]
    X = X_df.drop(columns=drop_candidates, errors='ignore')
    # select numeric columns
    X = X.select_dtypes(include=[np.number]).fillna(0.0)
    return X


def compute_ae_mse(ae_model, X):
    # X is numpy array
    recon = ae_model.predict(X)
    mse = np.mean(np.square(X - recon), axis=1)
    return mse


def fit_models_and_score(X_train, X_out, use_ae=False, ae_model_path=None):
    # scaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_out_s = scaler.transform(X_out)

    # IsolationForest
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(X_train_s)
    iso_scores = -iso.decision_function(X_out_s)  # invert: higher = more anomalous

    # LOF (novelty)
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.05)
    lof.fit(X_train_s)
    lof_scores = -lof.decision_function(X_out_s)

    # autoencoder mse
    ae_mse = None
    if use_ae and HAS_TF and ae_model_path and Path(ae_model_path).exists():
        try:
            ae = load_model(ae_model_path)
            ae_mse = compute_ae_mse(ae, X_out_s)
        except Exception:
            ae_mse = None

    return iso_scores, lof_scores, ae_mse


def make_ensemble(df_scores, score_cols):
    # Convert scores to ranks (higher => more anomalous)
    ranks = pd.DataFrame(index=df_scores.index)
    for c in score_cols:
        ranks[c] = df_scores[c].rank(method='average', pct=True)

    # rank-sum ensemble
    df_scores['ensemble_rank_mean'] = ranks.mean(axis=1)
    df_scores['ensemble_rank_sum'] = ranks.sum(axis=1) / len(score_cols)

    # zscore mean ensemble
    from scipy.stats import zscore
    zs = df_scores[score_cols].apply(lambda x: zscore(x, nan_policy='omit'))
    # for columns where higher means more anomalous, zscore is fine; ensure NaNs handled
    df_scores['ensemble_mean_z'] = zs.mean(axis=1)

    # Normalize ensemble_mean_z to 0..1 via min-max
    col = 'ensemble_mean_z'
    v = df_scores[col].values
    v = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-12)
    df_scores['ensemble_mean_z_norm'] = v

    # Ensure higher == more anomalous for all ensemble outputs
    return df_scores


def sweep_threshold_and_eval(scores, y_true, min_c, max_c, steps):
    rows = []
    n = len(scores)
    for c in np.linspace(min_c, max_c, steps):
        # threshold to flag top-c fraction as anomalies
        if c <= 0:
            pred = np.zeros(n, dtype=int)
        else:
            thr = np.quantile(scores, 1.0 - c)
            pred = (scores >= thr).astype(int)

        p, r, f, _ = precision_recall_fscore_support(y_true, pred, average='binary', zero_division=0)
        rows.append({'contamination': float(c), 'precision': float(p), 'recall': float(r), 'f1': float(f), 'pred_positives': int(pred.sum())})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='Notebook/results/anomaly_scores_by_user.csv')
    p.add_argument('--user_features', default='Notebook/user_features.csv')
    p.add_argument('--answers', default='Notebook/answers_r42.csv')
    p.add_argument('--ae_model', default='models/autoencoder.keras')
    p.add_argument('--min', type=float, default=0.001)
    p.add_argument('--max', type=float, default=0.5)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--folds', type=int, default=5)
    args = p.parse_args()

    out_dir = Path('Notebook/results')
    out_dir.mkdir(parents=True, exist_ok=True)

    df_scores_file, X_df, y_df = load_inputs(args.results, args.user_features, args.answers)

    # Merge features and answers on user id. answers likely have first col 'user'
    # Normalize column names
    if 'user' not in X_df.columns and 'user_id' in X_df.columns:
        X_df = X_df.rename(columns={'user_id': 'user'})

    if 'user' not in y_df.columns and y_df.shape[1] >= 1:
        y_df = y_df.rename(columns={y_df.columns[0]: 'user'})

    merged = X_df.merge(y_df[['user', 'label']], on='user', how='inner')
    if merged.empty:
        raise SystemExit('No overlap between user_features and answers; check user id columns')

    X_all = prepare_feature_matrix(merged)
    users = merged['user'].values
    y_all = merged['label'].astype(int).values

    # Outer holdout
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
    train_val_idx, test_idx = next(sss.split(X_all, y_all))

    X_train_val = X_all.iloc[train_val_idx].values
    y_train_val = y_all[train_val_idx]
    users_train_val = users[train_val_idx]

    X_test = X_all.iloc[test_idx].values
    y_test = y_all[test_idx]
    users_test = users[test_idx]

    fold_best_contams = []
    combined_rows = []

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    for i, (tr_idx, val_idx) in enumerate(skf.split(X_train_val, y_train_val), start=1):
        print(f'Fold {i}: training models and scoring validation set...')
        X_tr = X_train_val[tr_idx]
        X_val = X_train_val[val_idx]
        y_val = y_train_val[val_idx]
        users_val = users_train_val[val_idx]

        iso_val, lof_val, ae_val = fit_models_and_score(X_tr, X_val, use_ae=HAS_TF, ae_model_path=args.ae_model)

        dfv = pd.DataFrame({'user': users_val})
        dfv['iso_score'] = iso_val
        dfv['lof_score'] = lof_val
        if ae_val is not None:
            dfv['ae_mse'] = ae_val

        # build ensemble from available score cols
        score_cols = [c for c in ['iso_score', 'lof_score', 'ae_mse'] if c in dfv.columns]
        dfv = make_ensemble(dfv, score_cols)

        # choose which ensemble column to tune (rank_mean normalized 0..1)
        # prefer ensemble_rank_mean
        if 'ensemble_rank_mean' in dfv.columns:
            tune_scores = dfv['ensemble_rank_mean'].values
        else:
            tune_scores = dfv[score_cols[0]].values

        tune_df = sweep_threshold_and_eval(tune_scores, y_val, args.min, args.max, args.steps)
        tune_df.to_csv(out_dir / f'ensemble_threshold_tuning_fold{i}.csv', index=False)

        # pick best f1
        best_row = tune_df.loc[tune_df['f1'].idxmax()].to_dict()
        print(f'Fold {i} best: contamination={best_row["contamination"]:.4f} f1={best_row["f1"]:.4f}')
        fold_best_contams.append(float(best_row['contamination']))

        # save combined rows for inspection
        dfv['label'] = y_val
        dfv['fold'] = i
        combined_rows.append(dfv)

    # combine per-fold validation scores
    combined_val = pd.concat(combined_rows, ignore_index=True)
    combined_val.to_csv(out_dir / 'ensemble_threshold_tuning_cv_combined.csv', index=False)

    # choose median contamination
    median_contam = float(np.median(fold_best_contams))
    summary = {'fold_best_contaminations': fold_best_contams, 'median_contamination': median_contam}
    with open(out_dir / 'ensemble_selected_threshold_cv.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # retrain on entire train_val and score holdout test
    print('Retraining models on full inner train+val and scoring holdout test...')
    iso_test_s, lof_test_s, ae_test_s = fit_models_and_score(X_train_val, X_test, use_ae=HAS_TF, ae_model_path=args.ae_model)

    dft = pd.DataFrame({'user': users_test})
    dft['iso_score'] = iso_test_s
    dft['lof_score'] = lof_test_s
    if ae_test_s is not None:
        dft['ae_mse'] = ae_test_s

    score_cols = [c for c in ['iso_score', 'lof_score', 'ae_mse'] if c in dft.columns]
    dft = make_ensemble(dft, score_cols)

    # select ensemble column used earlier
    if 'ensemble_rank_mean' in dft.columns:
        final_scores = dft['ensemble_rank_mean'].values
    else:
        final_scores = dft[score_cols[0]].values

    # threshold at median_contam
    thr = np.quantile(final_scores, 1.0 - median_contam)
    preds = (final_scores >= thr).astype(int)

    print('Final test evaluation:')
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, zero_division=0))

    # save per-user ensemble scores and preds
    dft['ensemble_score'] = final_scores
    dft['pred'] = preds
    dft['label'] = y_test
    dft.to_csv(out_dir / 'ensemble_anomaly_scores_by_user.csv', index=False)

    print('Saved ensemble artifacts to Notebook/results/')


if __name__ == '__main__':
    main()
