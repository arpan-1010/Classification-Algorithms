import pandas as pd
import numpy as np

data = pd.read_csv('german_credit.csv')

X_features = list(data.columns)
X_features.remove('credit_rating')
print(X_features)

encoded_data = pd.get_dummies(data[X_features], drop_first = True)
print(list(encoded_data.columns))

print(encoded_data[['checking_account_status_A12', 'checking_account_status_A13', 'checking_account_status_A14']].head(5))

import statsmodels.api as sm

Y = data.credit_rating
X = sm.add_constant(encoded_data)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

logit = sm.Logit(y_train, X_train.astype(float))
logit_model = logit.fit()

print(logit_model.summary2())

def get_significant_vars( lm ) :
    var_p_vals_df = pd.DataFrame(lm.pvalues)
    var_p_vals_df['vars'] = var_p_vals_df.index
    var_p_vals_df.columns = ['pvals', 'vars']
    return list(var_p_vals_df[var_p_vals_df.pvals <= 0.05]['vars'])

significant_vars = get_significant_vars(logit_model)
#print(significant_vars)

final_logit = sm.Logit(y_train, sm.add_constant(X_train[significant_vars].astype(float))).fit()
print(final_logit.summary2())

y_pred_df = pd.DataFrame({"actual" : y_test, "predicted_prob" : final_logit.predict(sm.add_constant(X_test[significant_vars]).astype(float))})
print(y_pred_df.sample(10, random_state = 42))

y_pred_df['predicted'] = y_pred_df.predicted_prob.map(lambda x: 1 if x >= 0.5 else 0)
print(y_pred_df.sample(10, random_state = 42))

import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import metrics

def draw_cm(actual, predicted) :
    cm = metrics.confusion_matrix(actual, predicted)
    sn.heatmap(cm, annot=True, fmt = '.2f', xticklabels=['Bad Credit', 'Good Credit'], yticklabels=['Bad Credit', 'Good Credit'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

draw_cm(y_pred_df.actual, y_pred_df.predicted)

print(metrics.classification_report(y_pred_df.actual, y_pred_df.predicted))







