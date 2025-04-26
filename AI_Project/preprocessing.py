import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2

train_df = pd.read_csv('train_dataset.csv.csv')
test_df = pd.read_csv('test_dataset.csv.csv')

# 1. Handle Missing Values
def fill_missing_values(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(df[column].mean(), inplace=True)
    return df

train_df = fill_missing_values(train_df)
test_df = fill_missing_values(test_df)


# 2. Encode Categorical Features
binary_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
le = LabelEncoder()

for col in binary_cols:
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

multi_class_cols = ['CAEC', 'CALC', 'MTRANS']
train_df = pd.get_dummies(train_df, columns=multi_class_cols)
test_df = pd.get_dummies(test_df, columns=multi_class_cols)

train_cols = set(train_df.columns)
test_cols = set(test_df.columns)

missing_cols_in_test = train_cols - test_cols
for col in missing_cols_in_test:
    test_df[col] = 0

extra_cols_in_test = test_cols - train_cols
test_df.drop(columns=list(extra_cols_in_test), inplace=True)

test_df = test_df[train_df.columns.drop('NObeyesdad')]


# 3. Normalize Numerical Features
numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

scaler = MinMaxScaler()
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])


# 4. Feature Selection
X_train = train_df.drop('NObeyesdad', axis=1)
y_train = train_df['NObeyesdad']

X_test = test_df

selector = SelectKBest(score_func=chi2, k='all')
X_train_selected = selector.fit_transform(X_train, y_train)

selected_features = X_train.columns[selector.get_support()]

X_train = X_train[selected_features]
X_test = X_test[selected_features]

