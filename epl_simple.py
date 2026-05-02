import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import kagglehub
import os

# 1. Load Dataset
print("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("calvinrostanto/fantasy-premier-league-2025-2026")
csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(path, csv_file))

# 2. Data Preparation
NUMERIC_FEATURES = ['minutes', 'saves', 'clean_sheets', 'goals_scored', 'assists']
TARGET_COL = 'points_per_game'

df = df[NUMERIC_FEATURES + ['position_name', TARGET_COL]].dropna()

for f in NUMERIC_FEATURES:
    df[f] = df[f].fillna(df[f].median())

df['is_top_player'] = (df[TARGET_COL] >= 4.5).astype(int)

# One-Hot Encode Position
df_encoded = pd.get_dummies(df, columns=['position_name'], prefix='pos')
pos_columns = [col for col in df_encoded.columns if col.startswith('pos_')]
FEATURES = NUMERIC_FEATURES + pos_columns

X = df_encoded[FEATURES].values
y = df_encoded['is_top_player'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Build & Train Model (3 Rounds Protocol)
def build_model():
    model = Sequential([
        Input(shape=(len(FEATURES),)),
        Dense(8, activation='relu'),
        Dense(4, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

best_acc = 0
best_model = None
seeds = [42, 7, 2025]

print("\nTraining Position-Based Model (3 Rounds)...")
for seed in seeds:
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    model = build_model()
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    model.fit(X_train, y_train, validation_split=0.15, epochs=100, batch_size=16, callbacks=[early_stop], verbose=0)
    
    acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"Round (seed={seed}) - Accuracy: {acc*100:.2f}%")
    
    if acc > best_acc:
        best_acc = acc
        best_model = model

print(f"\nBest Model Accuracy: {best_acc*100:.2f}%")
print("----------------------------------------")

# 4. Interactive Prediction via Terminal
print("Choose Player Position:")
print("1: Goalkeeper")
print("2: Defender")
print("3: Midfielder")
print("4: Forward")
pos_choice = input("Enter choice (1-4): ")

pos_map = {'1': 'Goalkeeper', '2': 'Defender', '3': 'Midfielder', '4': 'Forward'}
selected_pos = pos_map.get(pos_choice, 'Midfielder')

print(f"\n--- Enter stats for {selected_pos} ---")
mins = float(input("Minutes Played : "))
saves = 0.0
cleans = 0.0
goals = 0.0
assists = 0.0

if selected_pos == 'Goalkeeper':
    saves = float(input("Saves : "))
    cleans = float(input("Clean Sheets : "))
elif selected_pos == 'Defender':
    cleans = float(input("Clean Sheets : "))
    goals = float(input("Goals Scored : "))
    assists = float(input("Assists : "))
elif selected_pos in ['Midfielder', 'Forward']:
    goals = float(input("Goals Scored : "))
    assists = float(input("Assists : "))

# Prepare data array matching FEATURES
data = {col: 0.0 for col in FEATURES}
data['minutes'] = mins
data['saves'] = saves
data['clean_sheets'] = cleans
data['goals_scored'] = goals
data['assists'] = assists

pos_col = f"pos_{selected_pos}"
if pos_col in data:
    data[pos_col] = 1.0

new_data = scaler.transform([[data[f] for f in FEATURES]])

prediction = best_model.predict(new_data, verbose=0)[0][0]

if prediction >= 0.5:
    print(f"\n=> Predicted Class: ELITE PLAYER (Confidence: {prediction*100:.2f}%)")
else:
    print(f"\n=> Predicted Class: AVERAGE PLAYER (Confidence: {(1-prediction)*100:.2f}%)")
