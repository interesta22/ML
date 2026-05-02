# 🏆 Premier League AI Scouting System
## Identifying Elite Players using Artificial Neural Networks

---

### 1️⃣ Problem Statement

**The Challenge**
In Fantasy Premier League (FPL), identifying which players will consistently deliver high points is the key to success. However, human bias and subjective scouting can often lead to poor decisions.

**The Solution**
We built a machine learning classification model using **Artificial Neural Networks (ANN)** to automatically evaluate a player's underlying statistics and classify them as an **Elite Player** or an **Average Player**. 

*Threshold:* We define an "Elite" player as one who consistently scores **≥ 4.5 points per game**.

---

### 2️⃣ The Dataset & Features

**Data Source**
We integrated the system directly with the live Kaggle dataset: `calvinrostanto/fantasy-premier-league-2025-2026`.

**Features Used for Prediction**
Instead of relying on luck, the model looks at hard data. The 6 core features fed into the network are:
1. `goals_scored`: Total goals
2. `assists`: Total assists
3. `minutes`: Total minutes played
4. `influence`: FPL index for game impact
5. `creativity`: FPL index for chance creation
6. `threat`: FPL index for goal threat

---

### 3️⃣ Artificial Neural Network Architecture

We built a deep learning model using TensorFlow and Keras, structured to learn complex patterns in player performance.

**Network Topology**
*   **Input Layer:** 6 Neurons (one for each feature)
*   **Hidden Layer 1:** 6 Neurons with `ReLU` activation (for non-linear pattern recognition)
*   **Hidden Layer 2:** 6 Neurons with `ReLU` activation
*   **Output Layer:** 1 Neuron with `Sigmoid` activation (outputs a probability between 0 and 100%)

*We used StandardScaler to normalize all features so the network learns evenly.*

---

### 4️⃣ The 3-Round Training Protocol

To ensure our model wasn't just getting lucky, we implemented a robust **3-Round Training Protocol**:

1.  **Multiple Seeds:** The model is initialized and trained 3 separate times using different random seeds (42, 7, 2025).
2.  **Early Stopping:** We stop training automatically if the validation loss doesn't improve for 15 epochs, preventing overfitting.
3.  **Selection:** The system evaluates all 3 rounds and **automatically selects the best performing model** for the final deployment.

---

### 5️⃣ Results & Evaluation

The best model is thoroughly evaluated on unseen test data to ensure it will perform well in real-world scouting.

**Key Metrics**
*   **Accuracy:** The percentage of correctly classified players.
*   **ROC-AUC Score:** Evaluates the model's ability to distinguish between the Elite and Average classes.

**Visual Diagnostics**
*   **Confusion Matrix:** Shows exactly how many False Positives (overhyped players) and False Negatives (hidden gems) the model predicted.
*   **ROC Curve:** Visually maps out the True Positive Rate against the False Positive Rate.

---

### 6️⃣ The Interactive AI Dashboard

To make the system usable by anyone (not just programmers), we built a beautiful, **EPL-themed Interactive Dashboard** directly inside the notebook using `ipywidgets`.

*   **Sliders:** Adjust the 6 features using intuitive sliders.
*   **Real-time Prediction:** Click the "PREDICT" button to run the data through the ANN in real-time.
*   **Confidence Score:** The dashboard provides a probability percentage and visually displays whether the player is an `ELITE PLAYER` 🏆 or an `AVERAGE PLAYER` 📊.

---
*Powered by TensorFlow, Python, and Data Science*
