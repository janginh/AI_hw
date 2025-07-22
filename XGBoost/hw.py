import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)

df = pd.read_csv("C:/Users/user/Desktop/수업관련/3학년 2025/1학기/인공지능/과제_장인환/XGBoost/Churn_Modelling.csv")
df.info()

# 불필요 정보 삭제
df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)

# 변수 인코딩(문자)
df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)
# x와 y 분리
X = df.drop("Exited", axis=1)
y = df["Exited"]
# 훈련/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# XGBoost 분류기 학습
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 평가 지표 출력
print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))
print(f"🔸 ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"🔸 PR AUC: {average_precision_score(y_test, y_proba):.4f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_test, y_proba):.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# PR Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure()
plt.plot(recall, precision, label=f"PR AUC = {average_precision_score(y_test, y_proba):.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.show()
