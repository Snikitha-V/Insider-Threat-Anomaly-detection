import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load features and answers (notebook paths)
features_path = os.path.join('Notebook','user_features.csv')
answers_path = os.path.join('Notebook','answers_r42.csv')
results_dir = os.path.join('Notebook','results')
os.makedirs(results_dir, exist_ok=True)

features_df = pd.read_csv(features_path)
answers_df = pd.read_csv(answers_path)

if 'user' in features_df.columns and 'user' in answers_df.columns:
    features_df['user'] = features_df['user'].astype(str).str.strip().str.upper()
    answers_df['user'] = answers_df['user'].astype(str).str.strip().str.upper()

if 'ground_truth' in answers_df.columns:
    features_df = features_df.merge(answers_df[['user','ground_truth']], on='user', how='left')
elif 'label' in answers_df.columns:
    features_df = features_df.merge(answers_df[['user','label']], on='user', how='left')
    features_df['ground_truth'] = features_df['label'].fillna(0).astype(int)
else:
    features_df['ground_truth'] = features_df['user'].isin(answers_df['user']).astype(int)

features_df['ground_truth'] = features_df['ground_truth'].fillna(0).astype(int)

# Build numeric X
drop_cols = ["user", "employee_name", "user_id", "anomaly_label", "anomaly_score", "label", "ground_truth"]
X = features_df.drop(columns=drop_cols, errors='ignore')
X = X.select_dtypes(include=[np.number]).fillna(0.0)

# split train/val
train_df, temp_df = train_test_split(features_df, test_size=0.3, random_state=42, stratify=features_df['ground_truth'])
val_df, test_df = train_test_split(temp_df, test_size=1/3, random_state=42, stratify=temp_df['ground_truth'])

X_train = train_df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).fillna(0.0)
X_val = val_df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).fillna(0.0)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

# Train lightweight autoencoder
try:
    from tensorflow.keras import layers, models, optimizers, callbacks
    input_dim = X_train_s.shape[1]
    encoding_dim = max(8, min(64, input_dim // 2))
    ae = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(encoding_dim, activation='relu'),
        layers.Dense(max(encoding_dim//2, 4), activation='relu'),
        layers.Dense(encoding_dim, activation='relu'),
        layers.Dense(input_dim, activation='linear'),
    ])
    ae.compile(optimizer=optimizers.Adam(1e-3), loss='mse')
    es = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ae.fit(X_train_s, X_train_s, validation_data=(X_val_s, X_val_s), epochs=200, batch_size=32, verbose=1, callbacks=[es])
    os.makedirs('models', exist_ok=True)
    ae.save('models/autoencoder.keras')
    print('Saved autoencoder to models/autoencoder.keras')
except Exception as e:
    print('Autoencoder training failed:', e)

print('Done')
