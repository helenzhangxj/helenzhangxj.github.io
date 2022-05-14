import requests
from oauth2client.service_account import ServiceAccountCredentials
import gspread

import json
import os
import gc
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 200)

# get file from gitlab
#  = input('token')
gitlab_url = f'http://gitlab.com/api/v4/projects/29240146/repository/files/gettickerdaily-61582555722a%2Ejson/raw?private_token=glpat-j62DEE-VAJyvAo4rqsnS'
response = requests.get(gitlab_url)
gc_key_file = response.json()

# define the scope
scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']

# add credentials to the account
creds = ServiceAccountCredentials.from_json_keyfile_dict(gc_key_file, scope)

# authorize the clientsheet 
client = gspread.authorize(creds)

# read data
sheet = client.open('rav_test')
sheet_instance = sheet.get_worksheet(0)
records_data = sheet_instance.get_all_records()
rav_df = pd.DataFrame.from_dict(records_data).replace('', np.nan)
rav_df.head()

# replace null to nan
rav_df = rav_df.fillna('nan')
rav_df.isna().sum()
rav_df.isnull().sum()

rav_df.describe().T

rav_df['jde_item_category_02_code'].value_counts()

# remove rows for target with count less than 5
target_cnt = rav_df['jde_item_category_02_code'].value_counts()
target_item_remove = target_cnt[target_cnt <=5].index.values
rav_df_subset = rav_df[~rav_df['jde_item_category_02_code'].isin(target_item_remove)]
rav_df_subset.shape

# split dataset to train, test
train_df, test_df = train_test_split(rav_df_subset, test_size = 0.2, random_state = 153)

# build model   

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

# separate features, target
feat_col = train_df.columns[[2,4,5]]
X, y = train_df[feat_col], train_df.iloc[:,0].values

# target label encoding

def preprocess_target(y):
  le = LabelEncoder()
  y_encode = le.fit_transform(y)

  return le, y_encode

le_target, y_prep = preprocess_target(y)

# knn
# train, validation
X_train, X_val, y_train, y_val = train_test_split(X, y_prep, test_size = 0.3, random_state = 48, stratify=y_prep)

# pipeline, data preprocessing (one hot encoding), model (knn)
clf_knn_pipe = Pipeline(steps=[('oe', OneHotEncoder(handle_unknown='ignore')),
                           ('knn_clf', KNeighborsClassifier())])

params = {'knn_clf__n_neighbors': [2,3,5,7], 
          'knn_clf__p': [1,2],
          'knn_clf__leaf_size': [1, 5, 10, 15, 30]}

knn_res = GridSearchCV(clf_knn_pipe, params, 
cv=KFold(5, shuffle = True, random_state=2), scoring='accuracy', 
verbose = 20).fit(X_train, y_train)

knn_res.best_score_
knn_res.best_estimator_

# best params n_neighbors = 3, leaf_size=1, p=1

clf_knn_pipe_final = Pipeline(steps=[('oe', OneHotEncoder(handle_unknown='ignore')),
                           ('knn_clf', KNeighborsClassifier(n_neighbors=3, leaf_size=1, p=1))])

clf_knn = clf_knn_pipe_final.fit(X_train, y_train)
y_pred = clf_knn.predict(X_val)
accuracy_score(y_val, y_pred)

# refit on the full dataset
clf_knn_final = clf_knn_pipe_final.fit(X, y_prep)


# predict on test dataset / new data (test_df)

X_test = test_df[feat_col]
y_test = test_df.iloc[:,0].values

y_output = le_target.inverse_transform(clf_knn_final.predict(X_test))
accuracy_score(y_test, y_output)


# SGD
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

# train, validation
X_train, X_val, y_train, y_val = train_test_split(X, y_prep, test_size = 0.3, random_state = 48, stratify=y_prep)

# pipeline, data preprocessing (one hot encoding), create feature, model (SGD)
clf_sgd_pipe = Pipeline(steps=[('oe', OneHotEncoder(handle_unknown='ignore')),
                           ('rbf_feature', RBFSampler(gamma=1, random_state=1)),
                           ('sgd_clf', SGDClassifier(max_iter=5))])

sgd_res = cross_val_score(clf_sgd_pipe, X_train, y_train, 
cv=KFold(5, shuffle = True, random_state=2), 
scoring = 'accuracy', verbose = 20)

sgd_res
np.mean(sgd_res)

clf_sgd = clf_sgd_pipe.fit(X_train, y_train)
y_pred = clf_sgd.predict(X_val)
accuracy_score(y_val, y_pred)

# predict on test dataset / new data (test_df)
# rav_df = rav_df.fillna('nan')

X_test = test_df[feat_col]
y_test = test_df.iloc[:,0].values

y_output = le_target.inverse_transform(clf_sgd.predict(X_test))
accuracy_score(y_test, y_output)






















































