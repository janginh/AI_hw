from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydot

# 데이터 불러오기
df = pd.read_csv("drug200.csv")

# 인코딩
le_sex = LabelEncoder()
le_bp = LabelEncoder()
le_chol = LabelEncoder()
le_drug = LabelEncoder()

df['Sex'] = le_sex.fit_transform(df['Sex'])
df['BP'] = le_bp.fit_transform(df['BP'])
df['Cholesterol'] = le_chol.fit_transform(df['Cholesterol'])
df['Drug'] = le_drug.fit_transform(df['Drug'])

# 입력/출력 분리
x = df.drop(columns='Drug')
y = df['Drug']

# 학습용/테스트용 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=42)

# 결정 트리 모델 훈련
dTreeAll = DecisionTreeClassifier(random_state=0)
dTreeAll.fit(x_train, y_train)

print("Train Score: {:.2f}".format(dTreeAll.score(x_train, y_train)))
print("Test Score : {:.2f}".format(dTreeAll.score(x_test, y_test)))


export_graphviz(
    dTreeAll,
    out_file="decisionTree0.dot",
    class_names=le_drug.classes_, 
    feature_names=x.columns,
    impurity=False,
    filled=True
)

(graph,) = pydot.graph_from_dot_file('decisionTree0.dot', encoding='utf8')
graph.write_png('decisionTree0.png')



# 1. 이진 분류용으로 y_test, y_pred_proba 준비 (Drug Y vs others)
# Drug Y는 클래스 4번에 해당하므로 → 1, 나머지는 0
y_test_binary = (y_test == 4).astype(int)
y_proba = dTreeAll.predict_proba(x_test)[:, 4]  # 클래스 4에 대한 확률만 추출

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test_binary, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Drug Y")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 3. PR Curve
precision, recall, _ = precision_recall_curve(y_test_binary, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label="PR Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Drug Y")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 4. AUC 값 출력
roc_auc_val = roc_auc_score(y_test_binary, y_proba)
print("ROC AUC Score (Drug Y vs others):", round(roc_auc_val, 3))
