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

import gradio as gr

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
EPL_GREEN   = "#00ff85"
EPL_BG      = "#1a0a2e"

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.patch.set_facecolor(EPL_BG)
fig.suptitle(" EPL 2025/2026 — Feature Distributions",
       color=EPL_GREEN, fontsize=18, fontweight='bold', y=1.02)

features_plot = ['minutes', 'goals_scored', 'assists', 'clean_sheets', 'saves', 'points_per_game']
colors_plot = [EPL_MAGENTA, EPL_GREEN, "#a855f7", EPL_MAGENTA, EPL_GREEN, "#a855f7"]

for ax, feat, col in zip(axes.flat, features_plot, colors_plot):
  ax.set_facecolor("#0d0621")
  ax.hist(df_raw[feat].dropna(), bins=30, color=col, edgecolor='none', alpha=0.85)
  ax.set_title(feat.replace('_', ' ').title(), color='white', fontsize=12, pad=8)
  ax.tick_params(colors='#aaa')
  for spine in ax.spines.values():
    spine.set_edgecolor('#2a1050')

plt.tight_layout()
plt.savefig("eda_distributions.png", dpi=150, bbox_inches='tight', facecolor=EPL_BG)
plt.close()
print(" EDA complete.")


# ─── STEP 3: Data Cleaning & Preprocessing ──────────────────
print("\n Data Cleaning...")

NUMERIC_FEATURES = ['minutes', 'saves', 'clean_sheets', 'goals_scored', 'assists']
TARGET_COL = 'points_per_game'

# Keep only required columns
df = df_raw[NUMERIC_FEATURES + ['position_name', TARGET_COL]].copy()

# Convert columns to numeric
for col in NUMERIC_FEATURES + [TARGET_COL]:
  df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop missing targets
before = len(df)
df.dropna(subset=[TARGET_COL, 'position_name'], inplace=True)
print(f"  Dropped {before - len(df)} rows with missing target or position.")

# Fill remaining NaN in features with median
for feat in NUMERIC_FEATURES:
  median_val = df[feat].median()
  df[feat].fillna(median_val, inplace=True)

# Create Target Label
THRESHOLD = 4.5
df['is_top_player'] = (df[TARGET_COL] >= THRESHOLD).astype(int)

# One-Hot Encode Position
df_encoded = pd.get_dummies(df, columns=['position_name'], prefix='pos')
pos_columns = [col for col in df_encoded.columns if col.startswith('pos_')]
FEATURES = NUMERIC_FEATURES + pos_columns

print(f"\n Target Distribution (threshold = {THRESHOLD} pts/game):")
class_counts = df_encoded['is_top_player'].value_counts()
print(f"  Elite (1): {class_counts.get(1, 0)} players ({class_counts.get(1, 0)/len(df)*100:.1f}%)")
print(f"  Average(0): {class_counts.get(0, 0)} players ({class_counts.get(0, 0)/len(df)*100:.1f}%)")


# ─── STEP 4: Feature/Target Split & Train-Test Split ────────
X = df_encoded[FEATURES].values
y = df_encoded['is_top_player'].values

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.20, random_state=42, stratify=y)

print(f"\n Data split complete:")
print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Test set:   {X_test.shape[0]} samples")
print(f"  Features:   {X_train.shape[1]}")


# ─── STEP 5: Feature Scaling (StandardScaler) ───────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)


# ─── STEP 6: Build ANN Model ─────────────────────────────────
def build_ann(input_dim: int) -> keras.Model:
  model = keras.Sequential([
    layers.Input(shape=(input_dim,), name="input_layer"),
    layers.Dense(8, activation='relu', name="hidden_layer_1"),
    layers.Dense(4, activation='relu', name="hidden_layer_2"),
    layers.Dense(1, activation='sigmoid', name="output_layer"),
  ], name="EPL_Position_ANN")

  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return model

model = build_ann(input_dim=len(FEATURES))
model.summary()


# ─── STEP 7: Training — 3 Rounds for Best Result ─────────────
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
  early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
  
  history = m.fit(X_train_sc, y_train, validation_split=0.15, epochs=EPOCHS, batch_size=16, callbacks=[early_stop], verbose=0)

  y_pred_prob = m.predict(X_test_sc, verbose=0).flatten()
  y_pred   = (y_pred_prob >= 0.5).astype(int)
  acc    = accuracy_score(y_test, y_pred)
  auc    = roc_auc_score(y_test, y_pred_prob)

  results.append({'round': rnd, 'accuracy': acc, 'auc': auc, 'model': m})
  histories.append(history)

  print(f"  Accuracy: {acc*100:.2f}% | AUC: {auc:.4f} | Epochs: {len(history.history['loss'])}")

  if acc > best_acc:
    best_acc  = acc
    best_model = m
    best_round = rnd

print(f"\n Best Model: Round {best_round} | Test Accuracy = {best_acc*100:.2f}%")


# ─── STEP 8: Learning Curves — All 3 Rounds ──────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.patch.set_facecolor(EPL_BG)
fig.suptitle(" Training History — 3 Rounds", color=EPL_GREEN, fontsize=16, fontweight='bold')

metrics = [('loss', 'val_loss'), ('accuracy', 'val_accuracy')]
titles = ['Loss', 'Accuracy']

for col, (rnd_info, hist) in enumerate(zip(results, histories)):
  for row, ((trn_key, val_key), title) in enumerate(zip(metrics, titles)):
    ax = axes[row][col]
    ax.set_facecolor("#0d0621")
    ax.plot(hist.history[trn_key],  color=EPL_MAGENTA, lw=2, label='Train')
    ax.plot(hist.history[val_key],  color=EPL_GREEN,  lw=2, label='Val', linestyle='--')
    ax.set_title(f"Round {rnd_info['round']} — {title}", color='white', fontsize=11)
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

print(f"\n Test Accuracy : {acc_final*100:.2f}%")
print(f" ROC-AUC Score : {auc_final:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_final)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(EPL_BG)

ax1 = axes[0]
ax1.set_facecolor("#0d0621")
sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', xticklabels=['Average', 'Elite'], yticklabels=['Average', 'Elite'], linewidths=0.5, linecolor='#2a1050', ax=ax1, annot_kws={"size": 14, "weight": "bold"})
ax1.set_title("Confusion Matrix", color='white', fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel('Predicted', color='#aaa')
ax1.set_ylabel('Actual', color='#aaa')
ax1.tick_params(colors='white')

ax2 = axes[1]
ax2.set_facecolor("#0d0621")
fpr, tpr, _ = roc_curve(y_test, y_pred_prob_final)
ax2.plot(fpr, tpr, color=EPL_GREEN, lw=2.5, label=f'AUC = {auc_final:.4f}')
ax2.plot([0, 1], [0, 1], color=EPL_MAGENTA, lw=1.5, linestyle='--', label='Random Baseline')
ax2.set_xlabel('False Positive Rate', color='#aaa')
ax2.set_ylabel('True Positive Rate', color='#aaa')
ax2.set_title("ROC Curve", color='white', fontsize=13, fontweight='bold', pad=12)
ax2.legend(facecolor='#1a0a2e', labelcolor='white')
ax2.tick_params(colors='#aaa')
for spine in ax2.spines.values():
  spine.set_edgecolor('#2a1050')

plt.tight_layout()
plt.savefig("evaluation_results.png", dpi=150, facecolor=EPL_BG)
plt.close()


# ─── STEP 10: Inter# ─── STEP 10: Premium Interactive Dashboard (Gradio) ──
import gradio as gr

logo_path = r"C:\Users\ahmed\.gemini\antigravity\brain\5ba6d1f0-3fd9-483a-8502-cbea03d6fdd5\epl_lion_logo_1777765971539.png"

epl_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700&display=swap');

:root {
    --epl-purple: #3d195b;
    --epl-magenta: #ff005a;
    --epl-green: #00ff85;
    --epl-navy: #020035;
}

body, .gradio-container { 
    background-color: var(--epl-navy) !important; 
    color: white !important; 
    font-family: 'Outfit', sans-serif !important; 
}

.main-box {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 30px;
    backdrop-filter: blur(10px);
}

.header-text {
    text-align: center;
    background: linear-gradient(90deg, var(--epl-green), var(--epl-magenta));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5em;
    font-weight: 800;
    margin-bottom: 20px;
}

.predict-btn {
    background: linear-gradient(45deg, var(--epl-magenta), #e91e63) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.1em !important;
    border-radius: 12px !important;
    transition: transform 0.2s !important;
}

.predict-btn:hover {
    transform: scale(1.02);
}

.output-verdict {
    background: rgba(0, 255, 133, 0.1);
    border: 1px solid var(--epl-green);
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    font-size: 1.5em;
    font-weight: 700;
    color: var(--epl-green) !important;
}

span, p, label { color: #ccc !important; font-weight: 500; }
"""

def predict_elite(pos, mins, saves, cleans, goals, assists):
    data = {col: 0.0 for col in FEATURES}
    data['minutes'] = float(mins)
    
    if pos == 'Goalkeeper':
        data['saves'] = float(saves)
        data['clean_sheets'] = float(cleans)
    elif pos == 'Defender':
        data['clean_sheets'] = float(cleans)
        data['goals_scored'] = float(goals)
        data['assists'] = float(assists)
    elif pos in ['Midfielder', 'Forward']:
        data['goals_scored'] = float(goals)
        data['assists'] = float(assists)
        
    pos_col = f"pos_{pos}"
    if pos_col in data:
        data[pos_col] = 1.0
        
    input_arr = np.array([[data[f] for f in FEATURES]])
    input_scaled = scaler.transform(input_arr)
    
    probability = float(best_model.predict(input_scaled, verbose=0).flatten()[0])
    
    if probability >= 0.5:
        return f"🏆 ELITE PLAYER IDENTIFIED\nScouting Confidence: {probability*100:.1f}%"
    else:
        return f"📊 AVERAGE PERFORMER\nScouting Confidence: {(1-probability)*100:.1f}%"

with gr.Blocks(css=epl_css, theme=gr.themes.Default()) as interface:
    with gr.Column(elem_classes="main-box"):
        with gr.Row():
            gr.Image(logo_path, show_label=False, width=120, container=False)
            gr.Markdown("# PREMIER LEAGUE\\nAI SCOUTING DASHBOARD", elem_classes="header-text")
            
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column():
                pos_dropdown = gr.Dropdown(
                    ['Goalkeeper', 'Defender', 'Midfielder', 'Forward'], 
                    label="PLAYER POSITION", 
                    value='Midfielder'
                )
                mins = gr.Slider(0, 3800, step=10, label="MINUTES PLAYED", value=1500)
                
            with gr.Column():
                saves = gr.Slider(0, 200, step=1, label="SAVES", visible=False)
                cleans = gr.Slider(0, 20, step=1, label="CLEAN SHEETS", visible=False)
                goals = gr.Slider(0, 35, step=1, label="GOALS SCORED", value=5)
                assists = gr.Slider(0, 25, step=1, label="ASSISTS", value=3)
        
        predict_btn = gr.Button("ANALYZE PERFORMANCE", elem_classes="predict-btn")
        
        gr.Markdown("### 🦁 Scouting Verdict")
        output_text = gr.Markdown(value="Waiting for input...", elem_classes="output-verdict")
        
    def update_ui(pos):
        if pos == 'Goalkeeper':
            return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
        elif pos == 'Defender':
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
        else:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
            
    pos_dropdown.change(fn=update_ui, inputs=pos_dropdown, outputs=[saves, cleans, goals, assists])
    predict_btn.click(fn=predict_elite, inputs=[pos_dropdown, mins, saves, cleans, goals, assists], outputs=output_text)

interface.launch(debug=True, share=True)
