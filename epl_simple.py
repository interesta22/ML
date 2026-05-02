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
dataset = pd.read_csv(os.path.join(path, csv_file))

# 2. Data Preparation
features = ['goals_scored', 'assists', 'minutes', 'influence', 'creativity', 'threat']
dataset = dataset[features + ['points_per_game']].dropna()

for f in features:
    dataset[f] = dataset[f].fillna(dataset[f].median())

X = dataset[features].values
y = (dataset['points_per_game'] >= 4.5).astype(int).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Build & Train Model (3 Rounds Protocol)
def build_model():
    model = Sequential([
        Input(shape=(6,)),
        Dense(6, activation='relu'),
        Dense(6, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

best_acc = 0
best_model = None
seeds = [42, 7, 2025]

print("\nTraining Model (3 Rounds)...")
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
print("Enter player stats to predict if they are Elite:")
goals = float(input("Goals Scored : "))
assists = float(input("Assists : "))
minutes = float(input("Minutes Played : "))
influence = float(input("Influence : "))
creativity = float(input("Creativity : "))
threat = float(input("Threat : "))

# Scale user input
new_data = scaler.transform([[goals, assists, minutes, influence, creativity, threat]])

# Predict
prediction = best_model.predict(new_data, verbose=0)[0][0]

if prediction >= 0.5:
    print(f"\n=> Predicted Class: ELITE PLAYER (Confidence: {prediction*100:.2f}%)")
else:
    print(f"\n=> Predicted Class: AVERAGE PLAYER (Confidence: {(1-prediction)*100:.2f}%)")
