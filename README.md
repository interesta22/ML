# FPL ANN Classification System
## Fantasy Premier League 2025/26 — خطة التنفيذ الكاملة

> **Dataset:** `calvinrostanto/fantasy-premier-league-2025-2026` via KaggleHub  
> **Framework:** TensorFlow / Keras  
> **Goal:** بناء 3 موديلات ANN للتصنيف تساعد في قرارات الـ FPL

---

## Project Overview

| Model | Target | Classes | Use Case |
|-------|--------|---------|----------|
| **Model A** | Points Tier | Star / Standard / Blank | اختيار الكابتن |
| **Model B** | Value for Money | Underpriced / Fairly priced / Overpriced | اكتشاف اللاعبين اللقطة |
| **Model C** | Clean Sheet Potential | High CS Chance / Low CS Chance | اختيار المدافعين والحراس |

---

## Step 1 — Environment Setup

```bash
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow keras-tuner
```

---

## Step 2 — Load Dataset

```python
import kagglehub
import os
import pandas as pd

path = kagglehub.dataset_download("calvinrostanto/fantasy-premier-league-2025-2026")
print("Path to dataset files:", path)
print(os.listdir(path))

# اقرأ الملف الرئيسي (اتحقق من اسمه الفعلي)
df = pd.read_csv(f"{path}/players.csv")
print(df.shape)
print(df.columns.tolist())
```

---

## Step 3 — Exploratory Data Analysis (EDA)

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Basic info
print(df.info())
print(df.describe())
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())

# Distribution of key columns
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
cols = ['total_points', 'now_cost', 'minutes', 'goals_scored', 'assists', 'ict_index']
for i, col in enumerate(cols):
    if col in df.columns:
        df[col].hist(ax=axes[i//3][i%3], bins=30)
        axes[i//3][i%3].set_title(col)
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(14, 10))
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', center=0)
plt.title("Feature Correlation Matrix")
plt.show()
```

---

## Step 4 — Feature Engineering & Label Creation

```python
# === تأكد إن الـ columns دي موجودة أو عدّل الأسماء ===
# الـ columns المتوقعة: total_points, now_cost, element_type, minutes,
# goals_scored, assists, bonus, bps, influence, creativity, threat, ict_index, form

# --- Model A: Points Tier ---
def points_tier(pts):
    if pts >= 8:   return "Star"
    elif pts >= 2: return "Standard"
    else:          return "Blank"

df['points_tier'] = df['total_points'].apply(points_tier)
print("Model A distribution:\n", df['points_tier'].value_counts())

# --- Model B: Value for Money ---
df['points_per_million'] = df['total_points'] / (df['now_cost'] / 10)

def value_label(row):
    pos_median = df.groupby('element_type')['points_per_million'].median().to_dict()
    threshold = pos_median.get(row['element_type'], df['points_per_million'].median())
    ratio = row['points_per_million'] / threshold if threshold > 0 else 1
    if ratio > 1.2:   return "Underpriced"
    elif ratio > 0.8: return "Fairly priced"
    else:             return "Overpriced"

df['value_label'] = df.apply(value_label, axis=1)
print("Model B distribution:\n", df['value_label'].value_counts())

# --- Model C: Clean Sheet Potential (GK=1, DEF=2) ---
defenders = df[df['element_type'].isin([1, 2])].copy()

# عدّل اسم الـ column لو مختلف في الداتا (مثلاً: fixture_difficulty, difficulty_rating)
difficulty_col = 'difficulty'  # <-- غيّر لو الاسم مختلف

if difficulty_col in defenders.columns:
    defenders['cs_label'] = defenders[difficulty_col].apply(
        lambda d: "High CS Chance" if d <= 2 else "Low CS Chance"
    )
else:
    # بديل: نستخدم clean_sheets لو موجود
    defenders['cs_label'] = (defenders['clean_sheets'] >= 3).map(
        {True: "High CS Chance", False: "Low CS Chance"}
    )

print("Model C distribution:\n", defenders['cs_label'].value_counts())
```

---

## Step 5 — Data Preprocessing & Cleaning

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Features لكل موديل — عدّل الأسماء حسب الداتا الفعلية
FEATURES_AB = [
    'minutes', 'goals_scored', 'assists', 'bonus', 'bps',
    'influence', 'creativity', 'threat', 'ict_index', 'form',
    'now_cost', 'selected_by_percent', 'transfers_in', 'transfers_out'
]

FEATURES_C = [
    'minutes', 'clean_sheets', 'goals_conceded', 'saves',
    'bonus', 'bps', 'influence', 'now_cost', 'form'
]

# فلتر الـ features الموجودة فعلاً
FEATURES_AB = [f for f in FEATURES_AB if f in df.columns]
FEATURES_C  = [f for f in FEATURES_C  if f in defenders.columns]

print("Features for A/B:", FEATURES_AB)
print("Features for C:", FEATURES_C)

# --- Outlier removal بـ IQR ---
def remove_outliers(data, cols, multiplier=1.5):
    df_clean = data.copy()
    for col in cols:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            mask = (df_clean[col] >= Q1 - multiplier*IQR) & (df_clean[col] <= Q3 + multiplier*IQR)
            df_clean = df_clean[mask]
    return df_clean

df_clean = remove_outliers(df, FEATURES_AB)
def_clean = remove_outliers(defenders, FEATURES_C)

print(f"After outlier removal — Main: {df_clean.shape}, Defenders: {def_clean.shape}")

# --- Impute missing values ---
imputer_AB = SimpleImputer(strategy='median')
imputer_C  = SimpleImputer(strategy='median')

X_AB = imputer_AB.fit_transform(df_clean[FEATURES_AB])
X_C  = imputer_C.fit_transform(def_clean[FEATURES_C])

# --- Encode labels ---
le_A = LabelEncoder()
le_B = LabelEncoder()
le_C = LabelEncoder()

y_A = le_A.fit_transform(df_clean['points_tier'])
y_B = le_B.fit_transform(df_clean['value_label'])
y_C = le_C.fit_transform(def_clean['cs_label'])

print("Classes A:", le_A.classes_)  # ['Blank' 'Standard' 'Star']
print("Classes B:", le_B.classes_)  # ['Fairly priced' 'Overpriced' 'Underpriced']
print("Classes C:", le_C.classes_)  # ['High CS Chance' 'Low CS Chance']

# --- Scale features ---
scaler_AB = StandardScaler()
scaler_C  = StandardScaler()

X_AB_scaled = scaler_AB.fit_transform(X_AB)
X_C_scaled  = scaler_C.fit_transform(X_C)
```

---

## Step 6 — Handle Class Imbalance + Train/Val/Test Split

```python
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def prepare_splits(X, y, use_smote=True, random_state=42):
    """
    Split: 70% Train / 15% Validation / 15% Test (stratified)
    Optional SMOTE oversampling على الـ training set فقط
    """
    # Split test أولاً
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=random_state
    )
    # Split val من الباقي (0.176 ≈ 15% من الـ original)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=random_state
    )

    if use_smote:
        sm = SMOTE(random_state=random_state, k_neighbors=min(5, min(sum(y_train==c) for c in set(y_train))-1))
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"After SMOTE — Train: {X_train.shape}")

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

# Prepare all 3 datasets
X_train_A, X_val_A, X_test_A, y_train_A, y_val_A, y_test_A = prepare_splits(X_AB_scaled, y_A)
X_train_B, X_val_B, X_test_B, y_train_B, y_val_B, y_test_B = prepare_splits(X_AB_scaled, y_B)
X_train_C, X_val_C, X_test_C, y_train_C, y_val_C, y_test_C = prepare_splits(X_C_scaled, y_C)
```

---

## Step 7 — Build ANN Model

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def build_ann(input_dim, num_classes, units=(128, 64, 32), dropout=0.3):
    """
    Architecture:
      Input → Dense(128) + BN + ReLU + Dropout
            → Dense(64)  + BN + ReLU + Dropout
            → Dense(32)  + BN + ReLU + Dropout
            → Output (Softmax لو multi-class / Sigmoid لو binary)

    Activation Strategy:
      - Hidden layers: ReLU (default) — سريع وبيحل vanishing gradient
      - Output (3 classes): Softmax — بيحول لـ probabilities تجمعها 1
      - Output (2 classes): Sigmoid — للـ binary classification
    """
    model = Sequential()
    for i, u in enumerate(units):
        if i == 0:
            model.add(Dense(u, input_dim=input_dim))
        else:
            model.add(Dense(u))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(Dropout(dropout if i < len(units)-1 else dropout * 0.7))

    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:
        model.add(Dense(num_classes, activation='softmax'))
        loss = 'sparse_categorical_crossentropy'

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=['accuracy']
    )
    return model

CALLBACKS = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

# Build models
model_A = build_ann(input_dim=X_train_A.shape[1], num_classes=3)
model_B = build_ann(input_dim=X_train_B.shape[1], num_classes=3)
model_C = build_ann(input_dim=X_train_C.shape[1], num_classes=2)

model_A.summary()
```

---

## Step 8 — Train Models

```python
def train_model(model, X_train, y_train, X_val, y_val, model_name, epochs=100, batch_size=32):
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=CALLBACKS,
        verbose=1
    )

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['accuracy'],     label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title(f'{model_name} — Accuracy')
    axes[0].legend()

    axes[1].plot(history.history['loss'],     label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title(f'{model_name} — Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    return history

history_A = train_model(model_A, X_train_A, y_train_A, X_val_A, y_val_A, "Model A — Points Tier")
history_B = train_model(model_B, X_train_B, y_train_B, X_val_B, y_val_B, "Model B — Value for Money")
history_C = train_model(model_C, X_train_C, y_train_C, X_val_C, y_val_C, "Model C — Clean Sheet")
```

---

## Step 9 — Evaluate Models

```python
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score
)

def evaluate_model(model, X_test, y_test, class_names, model_name, is_binary=False):
    print(f"\n{'='*50}")
    print(f"Evaluation: {model_name}")
    print(f"{'='*50}")

    y_prob = model.predict(X_test)

    if is_binary:
        y_pred = (y_prob > 0.5).astype(int).flatten()
        y_prob_for_auc = y_prob.flatten()
        auc = roc_auc_score(y_test, y_prob_for_auc)
    else:
        y_pred = y_prob.argmax(axis=1)
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr')

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='macro')

    print(f"Accuracy : {acc:.4f}  (Target: >= 0.85)")
    print(f"F1 Macro : {f1:.4f}  (Target: >= 0.80)")
    print(f"ROC-AUC  : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

    return {"accuracy": acc, "f1_macro": f1, "roc_auc": auc}

results_A = evaluate_model(model_A, X_test_A, y_test_A,
    class_names=le_A.classes_.tolist(), model_name="Model A — Points Tier")

results_B = evaluate_model(model_B, X_test_B, y_test_B,
    class_names=le_B.classes_.tolist(), model_name="Model B — Value for Money")

results_C = evaluate_model(model_C, X_test_C, y_test_C,
    class_names=le_C.classes_.tolist(), model_name="Model C — Clean Sheet", is_binary=True)

# Summary table
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
for name, res in [("Model A (Points Tier)", results_A),
                  ("Model B (Value for Money)", results_B),
                  ("Model C (Clean Sheet)", results_C)]:
    print(f"{name:30s} | Acc: {res['accuracy']:.3f} | F1: {res['f1_macro']:.3f} | AUC: {res['roc_auc']:.3f}")
```

---

## Step 10 — Hyperparameter Tuning (لو النتائج أقل من الـ target)

```python
# شغّل بس لو accuracy < 85% أو F1 < 80%
import keras_tuner as kt

def model_builder(hp):
    num_layers  = hp.Int('num_layers', min_value=2, max_value=4)
    dropout     = hp.Float('dropout', 0.1, 0.5, step=0.1)
    activation  = hp.Choice('activation', ['relu', 'leaky_relu'])
    lr          = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])

    model = Sequential()
    for i in range(num_layers):
        units = hp.Int(f'units_{i}', min_value=32, max_value=256, step=32)
        model.add(Dense(units))
        model.add(BatchNormalization())
        if activation == 'leaky_relu':
            model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
        else:
            model.add(tf.keras.layers.Activation('relu'))
        model.add(Dropout(dropout))

    model.add(Dense(3, activation='softmax'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# مثال للـ Model A
tuner = kt.RandomSearch(
    model_builder,
    objective='val_accuracy',
    max_trials=20,
    directory='kt_results',
    project_name='model_a_tuning'
)

tuner.search(X_train_A, y_train_A,
             validation_data=(X_val_A, y_val_A),
             epochs=50,
             callbacks=CALLBACKS)

best_model_A = tuner.get_best_models(num_models=1)[0]
tuner.results_summary()
```

---

## Step 11 — Predict New Player

```python
import numpy as np

def predict_player(player_stats: dict, verbose=True):
    """
    player_stats: dict بنفس الـ features المستخدمة في التدريب
    """
    # Model A & B
    player_df_AB = pd.DataFrame([player_stats])[FEATURES_AB]
    player_imp_AB = imputer_AB.transform(player_df_AB)
    player_sc_AB  = scaler_AB.transform(player_imp_AB)

    pred_A_prob = model_A.predict(player_sc_AB, verbose=0)
    pred_B_prob = model_B.predict(player_sc_AB, verbose=0)
    pred_A = le_A.inverse_transform([pred_A_prob.argmax()])[0]
    pred_B = le_B.inverse_transform([pred_B_prob.argmax()])[0]

    # Model C (لو اللاعب مدافع أو حارس)
    pred_C = "N/A (forwards/midfielders)"
    if player_stats.get('element_type') in [1, 2]:
        player_df_C = pd.DataFrame([player_stats])[FEATURES_C]
        player_imp_C = imputer_C.transform(player_df_C)
        player_sc_C  = scaler_C.transform(player_imp_C)
        pred_C_prob = model_C.predict(player_sc_C, verbose=0)
        pred_C = le_C.inverse_transform([(pred_C_prob > 0.5).astype(int).flatten()[0]])[0]

    if verbose:
        print(f"""
╔══════════════════════════════════════════════╗
║           PLAYER PREDICTION RESULTS          ║
╠══════════════════════════════════════════════╣
║  Model A (Points Tier)  : {pred_A:<18} ║
║  Confidence             : {pred_A_prob.max():.1%:<18} ║
╠══════════════════════════════════════════════╣
║  Model B (Value)        : {pred_B:<18} ║
║  Confidence             : {pred_B_prob.max():.1%:<18} ║
╠══════════════════════════════════════════════╣
║  Model C (Clean Sheet)  : {str(pred_C):<18} ║
╚══════════════════════════════════════════════╝
        """)

    return {"points_tier": pred_A, "value": pred_B, "clean_sheet": pred_C}

# مثال — اعدّل الأرقام حسب لاعب فعلي من الداتا
example_player = {
    'minutes': 2700, 'goals_scored': 8, 'assists': 5,
    'bonus': 20, 'bps': 350, 'influence': 800.0,
    'creativity': 600.0, 'threat': 900.0, 'ict_index': 300.0,
    'form': 7.5, 'now_cost': 90, 'selected_by_percent': 25.0,
    'transfers_in': 50000, 'transfers_out': 10000,
    'element_type': 3  # 1=GK, 2=DEF, 3=MID, 4=FWD
}

result = predict_player(example_player)
```

---

## Step 12 — Save Models

```python
import joblib

# Save models
model_A.save('model_A_points_tier.h5')
model_B.save('model_B_value_money.h5')
model_C.save('model_C_clean_sheet.h5')

# Save preprocessors (مهم جداً — نفس الـ scaler في الـ prediction)
joblib.dump(scaler_AB,  'scaler_AB.pkl')
joblib.dump(scaler_C,   'scaler_C.pkl')
joblib.dump(imputer_AB, 'imputer_AB.pkl')
joblib.dump(imputer_C,  'imputer_C.pkl')
joblib.dump(le_A, 'label_encoder_A.pkl')
joblib.dump(le_B, 'label_encoder_B.pkl')
joblib.dump(le_C, 'label_encoder_C.pkl')

print("All models and preprocessors saved!")
```

---

## Architecture Summary

```
Input Layer  →  N features (normalized via StandardScaler)
     ↓
Dense(128)  +  BatchNormalization  +  ReLU  +  Dropout(0.3)
     ↓
Dense(64)   +  BatchNormalization  +  ReLU  +  Dropout(0.3)
     ↓
Dense(32)   +  BatchNormalization  +  ReLU  +  Dropout(0.2)
     ↓
Output Layer
  • Model A & B: Dense(3)  + Softmax   → 3 classes
  • Model C:     Dense(1)  + Sigmoid   → binary
```

### Activation Functions Justification

| Layer | Function | Why |
|-------|----------|-----|
| Hidden 1,2,3 | **ReLU** | Solves vanishing gradient, fast convergence |
| Hidden (alt) | **LeakyReLU** | Use if dead neurons problem appears |
| Output (3-class) | **Softmax** | Probability distribution over 3 classes |
| Output (binary) | **Sigmoid** | Binary probability [0, 1] |
| Between layers | **BatchNorm** | Stabilizes training, acts as regularizer |

---

## Target Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Accuracy | **≥ 85%** | Overall correctness |
| F1 Macro | **≥ 80%** | Handles class imbalance fairly |
| ROC-AUC | **≥ 0.85** | Discrimination ability |

---

## Column Names to Verify

قبل ما تشغّل الكود، تأكد إن الـ columns دي موجودة في الداتا:

```python
# شغّل الكود ده الأول
required_cols = [
    'total_points', 'now_cost', 'element_type', 'minutes',
    'goals_scored', 'assists', 'bonus', 'bps',
    'influence', 'creativity', 'threat', 'ict_index',
    'form', 'clean_sheets', 'goals_conceded', 'saves',
    'selected_by_percent', 'transfers_in', 'transfers_out'
]
missing = [c for c in required_cols if c not in df.columns]
print("Missing columns:", missing)
print("Available columns:", df.columns.tolist())
```

---

## Notes

- **SMOTE** بيتطبق على الـ training set بس — مش على الـ validation أو الـ test
- **نفس الـ scaler** اللي اتعمل fit على الـ training بيتستخدم في الـ prediction
- لو الداتا صغيرة جداً (< 500 rows) قلّل الـ units ونزّل الـ dropout
- لو في **dead neurons** (loss ثابت): غيّر من ReLU لـ LeakyReLU
- لو في **overfitting** (train acc عالية وval acc منخفضة): زوّد الـ dropout أو نزّل الـ units
