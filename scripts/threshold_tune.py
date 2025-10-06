
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import os

# CLI arguments to control sweep range and resolution
parser = argparse.ArgumentParser(description='Threshold tuning for anomaly scores')
parser.add_argument('--min', type=float, default=0.005, help='Minimum contamination percentile (fraction)')
parser.add_argument('--max', type=float, default=0.2, help='Maximum contamination percentile (fraction)')
parser.add_argument('--steps', type=int, default=40, help='Number of steps between min and max')
args = parser.parse_args()

# Paths
features_path = 'user_features.csv'
results_path = os.path.join('results', 'anomaly_scores_by_user.csv')

if not os.path.exists(features_path):
    raise FileNotFoundError(features_path)
if not os.path.exists(results_path):
    raise FileNotFoundError(results_path)

# Load
features_df = pd.read_csv(features_path)
res = pd.read_csv(results_path)

# normalize user
if 'user' in features_df.columns and 'user' in res.columns:
    features_df['user'] = features_df['user'].astype(str).str.strip().str.upper()
    res['user'] = res['user'].astype(str).str.strip().str.upper()
    # If ground_truth is already present use it, otherwise compute it from insiders.csv if available
    if 'ground_truth' in features_df.columns:
        merged = features_df[['user','ground_truth']].merge(res, on='user', how='left')
    else:
        # try to compute ground_truth using insiders.csv in Notebook/ or project root
        insiders_path_candidates = [os.path.join('insiders.csv'), os.path.join('..','Notebook','insiders.csv'), os.path.join('Notebook','insiders.csv')]
        insiders = None
        for p in insiders_path_candidates:
            if os.path.exists(p):
                insiders = pd.read_csv(p)
                break
        if insiders is not None and 'user' in insiders.columns:
            features_df['user'] = features_df['user'].astype(str).str.strip().str.upper()
            insiders['user'] = insiders['user'].astype(str).str.strip().str.upper()
            features_df['ground_truth'] = features_df['user'].isin(insiders['user']).astype(int)
            merged = features_df[['user','ground_truth']].merge(res, on='user', how='left')
        else:
            # fallback to positional merge
            merged = pd.concat([features_df.reset_index(drop=True), res.reset_index(drop=True)], axis=1)
else:
    # attempt positional alignment
    merged = pd.concat([features_df.reset_index(drop=True), res.reset_index(drop=True)], axis=1)

# If ground_truth not present, try to compute from insiders.csv as a final fallback
if 'ground_truth' not in merged.columns:
    insiders_path_candidates = [os.path.join('insiders.csv'), os.path.join('..','Notebook','insiders.csv'), os.path.join('Notebook','insiders.csv')]
    insiders = None
    for p in insiders_path_candidates:
        if os.path.exists(p):
            insiders = pd.read_csv(p)
            break
    if insiders is not None and 'user' in insiders.columns and 'user' in merged.columns:
        merged['user'] = merged['user'].astype(str).str.strip().str.upper()
        insiders['user'] = insiders['user'].astype(str).str.strip().str.upper()
        merged['ground_truth'] = merged['user'].isin(insiders['user']).astype(int)

if 'ground_truth' not in merged.columns:
    raise KeyError('ground_truth column not found and could not be computed from insiders.csv')

y_true = merged['ground_truth'].astype(int)

if 'anomaly_score' not in merged.columns and 'anomaly_label' not in merged.columns:
    raise KeyError('No anomaly_score or anomaly_label in results file')

scores = merged['anomaly_score'] if 'anomaly_score' in merged.columns else None
labels_saved = merged['anomaly_label'] if 'anomaly_label' in merged.columns else None

# If labels_saved exists, also compute baseline metrics comparing saved labels to ground truth
if labels_saved is not None:
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, labels_saved.fillna(0).astype(int), average='binary', zero_division=0)
    print('Saved-labels baseline -> precision: {:.3f}, recall: {:.3f}, f1: {:.3f}, positives: {}'.format(prec, rec, f1, int(labels_saved.sum())))

# Sweep thresholds if scores exist
if scores is not None:
    scores = scores.astype(float)
    # Sweep contamination percentiles using CLI-configured range
    percs = np.linspace(args.min, args.max, args.steps)
    rows = []
    for p in percs:
        thresh = np.nanpercentile(scores.dropna(), 100*p)
        y_pred = (scores <= thresh).astype(int).fillna(0)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        rows.append({'contamination': p, 'threshold': float(thresh), 'precision': float(prec), 'recall': float(rec), 'f1': float(f1), 'predicted_positives': int(y_pred.sum())})
    df_tune = pd.DataFrame(rows)
    best_row = df_tune.loc[df_tune['f1'].idxmax()]
    print('\nBest F1 on sweep:')
    print(best_row.to_dict())

    # thresholds achieving recall >= 0.5 (if any)
    rec50 = df_tune[df_tune['recall'] >= 0.5]
    if not rec50.empty:
        print('\nThresholds with recall >= 0.5:')
        print(rec50.head())
    else:
        print('\nNo threshold achieved recall >= 0.5 in the sweep')

    os.makedirs('results', exist_ok=True)
    out_path = os.path.join('results','threshold_tuning.csv')
    df_tune.to_csv(out_path, index=False)
    print('\nSaved tuning table to', out_path)
else:
    print('No scores to sweep; only saved labels baseline computed.')
