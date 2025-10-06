import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

path = r"Notebook/results/anomaly_scores_by_user.csv"
df = pd.read_csv(path)
print('Loaded saved results shape:', df.shape)

# Prepare features and target same as notebook
drop_cols = ["user", "employee_name", "user_id", "anomaly_label", "anomaly_score", "label"]
X = df.drop(columns=drop_cols, errors='ignore')
X = X.select_dtypes(include=[np.number])
print('Feature columns loaded:', X.columns.tolist())

# target
if 'label' in df.columns:
    y = df['label'].astype(int)
    print('Using ground-truth label')
else:
    y = df['anomaly_label'].astype(int)
    print('Using anomaly_label as target (model vs model)')

# scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# train a fresh IsolationForest and predict
model = IsolationForest(n_estimators=1000, contamination=0.05, random_state=42)
model.fit(X_train)
y_pred = model.predict(X_test)
y_pred = np.array([1 if p==-1 else 0 for p in y_pred])

# metrics
cm = confusion_matrix(y_test, y_pred)
print('\nConfusion matrix (rows=true, cols=pred):\n', cm)

p, r, f1, sup = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
metrics_df = pd.DataFrame({'precision': p, 'recall': r, 'f1': f1, 'support': sup}, index=['class_0_normal','class_1_anomaly'])
print('\nPer-class metrics:\n', metrics_df)
print('\nMacro F1:', f1.mean(), 'Weighted F1:', (f1 * (sup / sup.sum())).sum())

print('\nTest set class distribution (true):\n', y_test.value_counts())
print('Test set class distribution (predicted):\n', pd.Series(y_pred).value_counts())

metrics_df.to_csv('Notebook/results/eval_metrics_by_class.csv')
print('Saved metrics to Notebook/results/eval_metrics_by_class.csv')
