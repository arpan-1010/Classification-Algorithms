import pandas as pd
data = pd.read_csv('german_credit.csv')

X_features = list(data.columns)
X_features.remove('credit_rating')
encoded_data = pd.get_dummies(data[X_features], drop_first = True)

Y = data['credit_rating']
X = encoded_data

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier
clf_tree = DecisionTreeClassifier(criterion = 'gini', max_depth = 3)
clf_tree.fit(X_train, Y_train)

tree_predict = clf_tree.predict(X_test)
print(tree_predict)

gini_node_1 = 1 - pow(491 / 700, 2) - pow(209 / 700, 2)
print(round(gini_node_1, 4))

import math
entropy_node_1 = -(491 / 700) * math.log2(491 / 700) - (200 / 700) * math.log2(200 / 700)
print(round(entropy_node_1, 2))