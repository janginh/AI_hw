import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("C:/Users/user/Desktop/수업관련/3학년 2025/1학기/인공지능/과제_장인환/ARM/store_data.csv", header=None)

# 각 행을 리스트로 변경
transactions = df.apply(lambda row: row.dropna().tolist(), axis=1).tolist()

# One-hot encoding
te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_data, columns=te.columns_)

# Apriori
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

# rule 생성
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules.sort_values(by="lift", ascending=False)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
