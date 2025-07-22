import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
import matplotlib.pyplot as plt

#데이터 호출
df = pd.read_excel("C:/Users/user/Desktop/수업관련/3학년 2025/1학기/인공지능/과제_장인환/RandomForest/archive/MainData.xlsx")

# 데이터 분리
X = df.drop("target", axis=1)  # 'target' 종속변수
y = df["target"]

# 훈련데이터 / 테스트 데이터 구분
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # 양성 클래스 확률


#print("🔹 Confusion Matrix:")
#print(confusion_matrix(y_test, y_pred))

# 성능 출력
print("\n🔹 Classification Report:")
print(classification_report(y_test, y_pred))

# ROC AUC & PR AUC 계산
roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)
print(f"🔸 ROC AUC: {roc_auc:.4f}")
print(f"🔸 PR AUC: {pr_auc:.4f}")

# ROC Curve 
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# PR Curve 
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure()
plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.show()