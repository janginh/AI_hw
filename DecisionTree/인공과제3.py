from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#from IPython.display import Image
import pandas as pd
import pydot

df = pd.read_csv("drug200.csv")

# Label Encoding
le_sex = LabelEncoder()
le_bp = LabelEncoder()
le_chol = LabelEncoder()
le_drug = LabelEncoder()

df['Sex'] = le_sex.fit_transform(df['Sex'])           # 'F', 'M' → 0, 1
df['BP'] = le_bp.fit_transform(df['BP'])              # 'HIGH', 'NORMAL', 'LOW' → 0, 1, 2
df['Cholesterol'] = le_chol.fit_transform(df['Cholesterol'])
df['Drug'] = le_drug.fit_transform(df['Drug'])        # 'A'~'Y' → 0~4

x = df.drop(columns='Drug')
y = df['Drug']

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=42)

dTreeAll = DecisionTreeClassifier(random_state=0)
dTreeAll.fit(x_train, y_train)

print("Train Score: {:.2f}".format(dTreeAll.score(x_train, y_train)))
print("Test Score : {:.2f}".format(dTreeAll.score(x_test, y_test)))


export_graphviz(dTreeAll, out_file="dicisionTree0.dot", class_names =["malignant", "benign"],
    feature_names=df.feature.names, impurity=False, filled=True)                                                                     ])

(grapth,) = pydot.graph_from_dot_file('dicisionTree0.dot', encoding='utf8')

grapth.write_png('dicisionTree0.png')

'''
export_graphviz(
    dTreeAll,                     # 학습된 트리 모델
    out_file="tree.dot",
    feature_names=x.columns,
    class_names=le_drug.classes_,
    filled=True,
    rounded=True,
    special_characters=True
)

# pydot으로 그래프 읽기
(graph,) = pydot.graph_from_dot_file("tree.dot")

# .png 이미지로 저장
graph.write_png("tree.png")

# Jupyter Notebook에 그림으로 표시
Image(filename="tree.png")
'''