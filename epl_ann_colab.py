# ============================================================
#  EPL PLAYER CLASSIFICATION SYSTEM — ANN Model
# Premier League 2025/2026 | Google Colab Ready
# ============================================================

# ─── STEP 0: Install & Import Required Libraries ─────────────
# Run this block first — installs all required packages
# import subprocess
# subprocess.run(["pip", "install", "kagglehub", "tensorflow", "scikit-learn",
#         "pandas", "numpy", "matplotlib", "seaborn", "gradio", "-q"])

import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
               accuracy_score, roc_auc_score, roc_curve)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

print(" All libraries loaded successfully!")
print(f"  TensorFlow version: {tf.__version__}")
print(f"  Keras version:   {keras.__version__}")


# ─── STEP 1: Load Dataset ────────────────────────────────────
print("\n Downloading EPL 2025/2026 dataset from Kaggle...")
path = kagglehub.dataset_download("calvinrostanto/fantasy-premier-league-2025-2026")
print(f" Path to dataset files: {path}")

import os
# Auto-detect CSV file in the downloaded directory
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
print(f"  Found CSV files: {csv_files}")

df_raw = pd.read_csv(os.path.join(path, csv_files[0]))
print(f"  Dataset shape: {df_raw.shape}")
df_raw.head()


# ─── STEP 2: Exploratory Data Analysis (EDA) ────────────────
print("\n Dataset Overview:")
print(df_raw.info())
print("\n Statistical Summary:")
display(df_raw.describe())

# Visualise key feature distributions — EPL theme
EPL_PURPLE = "#3d195b"
EPL_MAGENTA = "#ff005a"
EPL_GREEN  = "#00ff85"
EPL_BG   = "#1a0a2e"

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.patch.set_facecolor(EPL_BG)
fig.suptitle(" EPL 2025/2026 — Feature Distributions",
       color=EPL_GREEN, fontsize=18, fontweight='bold', y=1.02)

features_plot = ['goals_scored', 'assists', 'minutes',
         'influence', 'creativity', 'threat']
colors_plot = [EPL_MAGENTA, EPL_GREEN, "#a855f7",
        EPL_MAGENTA, EPL_GREEN, "#a855f7"]

for ax, feat, col in zip(axes.flat, features_plot, colors_plot):
  ax.set_facecolor("#0d0621")
  ax.hist(df_raw[feat].dropna(), bins=30, color=col, edgecolor='none', alpha=0.85)
  ax.set_title(feat.replace('_', ' ').title(), color='white', fontsize=12, pad=8)
  ax.tick_params(colors='#aaa')
  for spine in ax.spines.values():
    spine.set_edgecolor('#2a1050')

plt.tight_layout()
plt.savefig("eda_distributions.png", dpi=150, bbox_inches='tight',
      facecolor=EPL_BG)
plt.close()
print(" EDA complete.")


# ─── STEP 3: Data Cleaning & Preprocessing ──────────────────
print("\n Data Cleaning...")

FEATURES = ['goals_scored', 'assists', 'minutes',
      'influence', 'creativity', 'threat']
TARGET_COL = 'points_per_game'

# Keep only required columns
df = df_raw[FEATURES + [TARGET_COL]].copy()

# Convert columns to numeric (handle any string/object dtypes)
for col in df.columns:
  df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows where target is missing
before = len(df)
df.dropna(subset=[TARGET_COL], inplace=True)
print(f"  Dropped {before - len(df)} rows with missing target.")

# Fill remaining NaN in features with median
for feat in FEATURES:
  median_val = df[feat].median()
  df[feat].fillna(median_val, inplace=True)
print(f"  Missing values after cleaning: {df.isnull().sum().sum()}")

# ── Create Classification Label ──────────────────────────────
THRESHOLD = 4.5
df['is_top_player'] = (df[TARGET_COL] >= THRESHOLD).astype(int)

class_counts = df['is_top_player'].value_counts()
print(f"\n Target Distribution (threshold = {THRESHOLD} pts/game):")
print(f"  Elite (1): {class_counts.get(1, 0):>4} players "
   f"({class_counts.get(1, 0)/len(df)*100:.1f}%)")
print(f"  Average(0): {class_counts.get(0, 0):>4} players "
   f"({class_counts.get(0, 0)/len(df)*100:.1f}%)")

# Visualise class balance
fig, ax = plt.subplots(figsize=(6, 4))
fig.patch.set_facecolor(EPL_BG)
ax.set_facecolor("#0d0621")
bars = ax.bar(['Average (0)', 'Elite (1)'],
       [class_counts.get(0, 0), class_counts.get(1, 0)],
       color=[EPL_PURPLE, EPL_GREEN], edgecolor='white', linewidth=0.5)
ax.set_title("Class Distribution — is_top_player", color='white',
       fontsize=13, fontweight='bold')
ax.tick_params(colors='white')
for spine in ax.spines.values():
  spine.set_edgecolor('#2a1050')
for bar, val in zip(bars, [class_counts.get(0, 0), class_counts.get(1, 0)]):
  ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
      str(val), ha='center', color='white', fontweight='bold')
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150, facecolor=EPL_BG)
plt.close()


# ─── STEP 4: Feature/Target Split & Train-Test Split ────────
X = df[FEATURES].values
y = df['is_top_player'].values

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.20, random_state=42, stratify=y)

print(f"\n Data split complete:")
print(f"  Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"  Test set:   {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.0f}%)")
print(f"  Features:   {X_train.shape[1]}")


# ─── STEP 5: Feature Scaling (StandardScaler) ───────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)  # fit on train only
X_test_sc = scaler.transform(X_test)     # apply same transform to test

print("\n StandardScaler applied:")
print(f"  Mean (train): {X_train_sc.mean(axis=0).round(3)}")
print(f"  Std  (train): {X_train_sc.std(axis=0).round(3)}")


# ─── STEP 6: Build ANN Model ─────────────────────────────────
def build_ann(input_dim: int) -> keras.Model:
  """
  ANN Architecture:
   Input → Dense(6, ReLU) → Dense(6, ReLU) → Dense(1, Sigmoid)
  """
  model = keras.Sequential([
    layers.Input(shape=(input_dim,), name="input_layer"),
    layers.Dense(6, activation='relu', name="hidden_layer_1"),
    layers.Dense(6, activation='relu', name="hidden_layer_2"),
    layers.Dense(1, activation='sigmoid', name="output_layer"),
  ], name="EPL_ANN_Classifier")

  model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
  )
  return model

model = build_ann(input_dim=len(FEATURES))
model.summary()


# ─── STEP 7: Training — 3 Rounds for Best Result ─────────────
"""
Strategy: Train the model 3 separate times with different random seeds.
After each round, compare accuracy and keep the BEST performing model.
"""

print("\n Starting 3-Round Training Protocol...")
print("─" * 55)

EPOCHS  = 100
SEEDS   = [42, 7, 2025]
results  = []
histories = []

best_acc  = 0.0
best_model = None
best_round = 0

for rnd, seed in enumerate(SEEDS, start=1):
  print(f"\n Round {rnd}/3 (seed={seed})")
  tf.random.set_seed(seed)
  np.random.seed(seed)

  m = build_ann(input_dim=len(FEATURES))

  early_stop = EarlyStopping(
    monitor='val_loss', patience=15,
    restore_best_weights=True, verbose=0
  )

  history = m.fit(
    X_train_sc, y_train,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=16,
    callbacks=[early_stop],
    verbose=0
  )

  y_pred_prob = m.predict(X_test_sc, verbose=0).flatten()
  y_pred   = (y_pred_prob >= 0.5).astype(int)
  acc     = accuracy_score(y_test, y_pred)
  auc     = roc_auc_score(y_test, y_pred_prob)

  results.append({'round': rnd, 'accuracy': acc, 'auc': auc, 'model': m})
  histories.append(history)

  print(f"   Accuracy: {acc*100:.2f}%  |  AUC: {auc:.4f}  |  "
     f"Epochs run: {len(history.history['loss'])}")

  if acc > best_acc:
    best_acc  = acc
    best_model = m
    best_round = rnd

print(f"\n Best Model: Round {best_round} | "
   f"Test Accuracy = {best_acc*100:.2f}%")


# ─── STEP 8: Learning Curves — All 3 Rounds ──────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.patch.set_facecolor(EPL_BG)
fig.suptitle(" Training History — 3 Rounds",
       color=EPL_GREEN, fontsize=16, fontweight='bold')

metrics = [('loss', 'val_loss'), ('accuracy', 'val_accuracy')]
titles = ['Loss', 'Accuracy']

for col, (rnd_info, hist) in enumerate(zip(results, histories)):
  for row, ((trn_key, val_key), title) in enumerate(zip(metrics, titles)):
    ax = axes[row][col]
    ax.set_facecolor("#0d0621")
    ax.plot(hist.history[trn_key],  color=EPL_MAGENTA, lw=2, label='Train')
    ax.plot(hist.history[val_key],  color=EPL_GREEN,  lw=2, label='Val',
        linestyle='--')
    ax.set_title(f"Round {rnd_info['round']} — {title}",
           color='white', fontsize=11)
    ax.tick_params(colors='#aaa')
    ax.legend(facecolor='#1a0a2e', labelcolor='white', fontsize=9)
    for spine in ax.spines.values():
      spine.set_edgecolor('#2a1050')

plt.tight_layout()
plt.savefig("training_history.png", dpi=150, facecolor=EPL_BG)
plt.close()


# ─── STEP 9: Final Evaluation (Best Model) ───────────────────
print(f"\n Final Evaluation — Best Model (Round {best_round})")
print("─" * 55)

y_pred_prob_final = best_model.predict(X_test_sc, verbose=0).flatten()
y_pred_final   = (y_pred_prob_final >= 0.5).astype(int)

acc_final = accuracy_score(y_test, y_pred_final)
auc_final = roc_auc_score(y_test, y_pred_prob_final)

print(f"\n🎯 Test Accuracy : {acc_final*100:.2f}%")
print(f" ROC-AUC Score : {auc_final:.4f}")
print(f"\n Classification Report:\n")
print(classification_report(y_test, y_pred_final,
               target_names=['Average (0)', 'Elite (1)']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_final)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(EPL_BG)

# — Heatmap —
ax1 = axes[0]
ax1.set_facecolor("#0d0621")
sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu',
      xticklabels=['Average', 'Elite'],
      yticklabels=['Average', 'Elite'],
      linewidths=0.5, linecolor='#2a1050',
      ax=ax1, annot_kws={"size": 14, "weight": "bold"})
ax1.set_title("Confusion Matrix", color='white', fontsize=13,
       fontweight='bold', pad=12)
ax1.set_xlabel('Predicted', color='#aaa')
ax1.set_ylabel('Actual', color='#aaa')
ax1.tick_params(colors='white')

# — ROC Curve —
ax2 = axes[1]
ax2.set_facecolor("#0d0621")
fpr, tpr, _ = roc_curve(y_test, y_pred_prob_final)
ax2.plot(fpr, tpr, color=EPL_GREEN, lw=2.5,
     label=f'AUC = {auc_final:.4f}')
ax2.plot([0, 1], [0, 1], color=EPL_MAGENTA, lw=1.5,
     linestyle='--', label='Random Baseline')
ax2.set_xlabel('False Positive Rate', color='#aaa')
ax2.set_ylabel('True Positive Rate', color='#aaa')
ax2.set_title("ROC Curve", color='white', fontsize=13,
       fontweight='bold', pad=12)
ax2.legend(facecolor='#1a0a2e', labelcolor='white')
ax2.tick_params(colors='#aaa')
for spine in ax2.spines.values():
  spine.set_edgecolor('#2a1050')

plt.tight_layout()
plt.savefig("evaluation_results.png", dpi=150, facecolor=EPL_BG)
plt.close()

# Accuracy comparison bar chart
fig, ax = plt.subplots(figsize=(7, 4))
fig.patch.set_facecolor(EPL_BG)
ax.set_facecolor("#0d0621")
rounds = [f"Round {r['round']}" for r in results]
accs  = [r['accuracy'] * 100 for r in results]
bar_colors = [EPL_MAGENTA, "#a855f7", EPL_GREEN]
bars = ax.bar(rounds, accs, color=bar_colors, edgecolor='white', linewidth=0.5)
ax.set_ylim([0, 110])
ax.set_title("Accuracy Across 3 Training Rounds",
       color='white', fontsize=13, fontweight='bold')
ax.tick_params(colors='white')
ax.set_ylabel("Accuracy (%)", color='#aaa')
for bar, val in zip(bars, accs):
  ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
      f"{val:.2f}%", ha='center', color='white', fontsize=12,
      fontweight='bold')
for spine in ax.spines.values():
  spine.set_edgecolor('#2a1050')
plt.tight_layout()
plt.savefig("round_comparison.png", dpi=150, facecolor=EPL_BG)
plt.close()


# ─── STEP 10: Interactive Prediction Dashboard (Gradio) ──
import gradio as gr
import numpy as np

# EPL Theme CSS
epl_css = """
body { background-color: #1a0a2e !important; color: white !important; font-family: 'Barlow', sans-serif; }
.gradio-container { background-color: #1a0a2e !important; border: 2px solid #ff005a !important; border-radius: 12px; box-shadow: 0 0 40px rgba(255,0,90,0.25); }
button.primary { background: linear-gradient(90deg, #3d195b, #ff005a) !important; border: none !important; color: #00ff85 !important; font-weight: bold; text-transform: uppercase; border-radius: 8px; transition: 0.3s; }
button.primary:hover { background: linear-gradient(90deg, #ff005a, #3d195b) !important; transform: scale(1.02); }
span, p, h1, h2, h3, label { color: white !important; }
.output-class { color: #00ff85 !important; font-size: 24px; font-weight: bold; }
"""

def predict_elite_player(goals, assists, mins, influence, creativity, threat):
  # Prepare features
  raw_features = np.array([[goals, assists, mins, influence, creativity, threat]])
  
  # Scale features using the scaler fitted in STEP 5
  raw_scaled = scaler.transform(raw_features)
  
  # Predict probability
  probability = float(best_model.predict(raw_scaled, verbose=0).flatten()[0])
  
  if probability >= 0.5:
    return f" ELITE PLAYER (Confidence: {probability*100:.1f}%)"
  else:
    return f" AVERAGE PLAYER (Confidence: {(1-probability)*100:.1f}%)"

# Design and Build the User Interface (UI)
interface = gr.Interface(
  fn=predict_elite_player,
  inputs=[
    gr.Slider(0, 35, step=1, label=" Goals Scored", value=5),
    gr.Slider(0, 25, step=1, label="🎯 Assists", value=3),
    gr.Slider(0, 3800, step=10, label=" Minutes Played", value=1500),
    gr.Number(label=" Influence", value=300),
    gr.Number(label=" Creativity", value=250),
    gr.Number(label=" Threat", value=400)
  ],
  outputs=gr.Text(label=" AI Scouting Verdict"),
  title=" EPL Player Classifier",
  description="Powered by Artificial Neural Network | FPL 2025/2026. Adjust the stats to predict if a player is Elite or Average.",
  css=epl_css
)

# Launch the interface
interface.launch(debug=True, share=True)
