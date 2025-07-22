import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import tensorflow as tf

# 데이터 불러오기
df = pd.read_csv("weatherAUS.csv")
df = df.drop(columns=["Evaporation", "Sunshine", "Cloud9am", "Cloud3pm"])
df = df.dropna(subset=["RainTomorrow"])
df[df.select_dtypes(include="float64").columns] = df.select_dtypes(include="float64").fillna(df.mean(numeric_only=True))

categorical_cols = df.select_dtypes(include=["object"]).columns.drop(["Date", "RainTomorrow"])
df = pd.get_dummies(df, columns=categorical_cols)
df["RainTomorrow"] = df["RainTomorrow"].map({"No": 0, "Yes": 1})
df = df.drop(columns=["Date"])

X = df.drop(columns=["RainTomorrow"]).values
y = df["RainTomorrow"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_cnn = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1, 1))

X_train, X_test, y_train, y_test = train_test_split(X_cnn, y, test_size=0.2, random_state=42, stratify=y)

# CNN 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 1), activation='relu', input_shape=(X_train.shape[1], 1, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
    tf.keras.layers.Conv2D(32, (3, 1), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2)

# 예측 및 성능 평가
y_probs = model.predict(X_test).ravel()
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

precision, recall, _ = precision_recall_curve(y_test, y_probs)
pr_auc = average_precision_score(y_test, y_probs)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(recall, precision, label=f"PR (AUC = {pr_auc:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.show()
