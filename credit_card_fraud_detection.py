print(__doc__)
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc


def prepared_df():
    # load the data
    df = pd.read_csv("creditcard.csv")

    # get column names
    col_names = df.columns.values
    print(col_names)

    # shape of dataset
    print("Shape of dataset:", df.shape)

    return df


def handle_imbalanced_class(df):
    # total number of entries in each class
    print("No. of Normal transaction:", df['Class'][df['Class'] == 0].count())
    print("No. of Fraudulent transaction:", df['Class'][df['Class'] == 1].count())

    # class seperation and randomization
    normal_transaction = df.query('Class == 0').sample(frac=1)
    fraud_transaction = df.query('Class == 1').sample(frac=1)

    # randomize the datasets
    # class_0 = normal_transaction.sample(frac=1)
    # class_1 = class_1.sample(frac=1)

    # undersample majority class
    normal_transaction_train = normal_transaction.iloc[0:10000]
    fraud_transaction_train = fraud_transaction

    # combine subset of different classes into one balaced dataframe
    combined_df = normal_transaction_train.append(fraud_transaction_train, ignore_index=True).values

    return combined_df


def train_test_xgboost(combined_df):
    # split data into x and y
    x = combined_df[:, 0:30].astype(float)
    y = combined_df[:, 30]

    model = XGBClassifier()
    kfold = StratifiedKFold(n_splits=10, random_state=7)

    # use area under the precision-recall curve to show classification accuracy
    scoring = 'roc_auc'
    results = cross_val_score(model, x, y, cv=kfold, scoring=scoring)
    print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))


def main():
    print("python main function")

    df = prepared_df()

    print("Data prepared!")

    combined_df = handle_imbalanced_class(df)

    print("Imbalanced class distribution handled!")

    train_test_xgboost(combined_df)


if __name__ == '__main__':
    main()