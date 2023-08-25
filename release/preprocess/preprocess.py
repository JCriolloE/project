import os
import sys

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from utilities import custom_logging

logger = custom_logging.CustomLogging(__name__)

SEED_SPLIT = 42
TARGET = 'Class'

class Preprocessing:

    def __init__(self, model_path):
        self.model_path = model_path

    def robustScaler(df):
        """
            RobustScaler is less prone to outliers.
        """
        rob_scaler = RobustScaler()

        df['scaled_amount'] = rob_scaler.fit_transform(
            df['Amount'].values.reshape(-1, 1))
        df['scaled_time'] = rob_scaler.fit_transform(
            df['Time'].values.reshape(-1, 1))

        df.drop(['Time', 'Amount'], axis=1, inplace=True)

        """
            Amount and Time are scaled
        """
        scaled_amount = df['scaled_amount']
        scaled_time = df['scaled_time']

        df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
        df.insert(0, 'scaled_amount', scaled_amount)
        df.insert(1, 'scaled_time', scaled_time)

        """
        Splitting the Data
        """
        X = df.drop(TARGET, axis=1)
        y = df[TARGET]

        sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

        for train_index, test_index in sss.split(X, y):
            print("Train:", train_index, "Test:", test_index)
            original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
            original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

        # Check the Distribution of the labels

        # Turn into an array
        original_Xtrain = original_Xtrain.values
        original_Xtest = original_Xtest.values
        original_ytrain = original_ytrain.values
        original_ytest = original_ytest.values

        # See if both the train and test label distribution are similarly
        # distributed
        train_unique_label, train_counts_label = np.unique(
            original_ytrain, return_counts=True)
        test_unique_label, test_counts_label = np.unique(
            original_ytest, return_counts=True)

        """
        Since our classes are highly skewed we should make them equivalent in order
        to have a normal distribution of the classes.
        Lets shuffle the data before creating the subsamples
        """

        df = df.sample(frac=1)

        # amount of fraud classes 492 rows.
        fraud_df = df.loc[df[TARGET] == 1]
        non_fraud_df = df.loc[df[TARGET] == 0][:492]

        normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

        # Shuffle dataframe rows
        new_df = normal_distributed_df.sample(frac=1, random_state=SEED_SPLIT)

        # # -----> V14 Removing Outliers (Highest Negative Correlated with Labels)
        v14_fraud = new_df['V14'].loc[new_df[TARGET] == 1].values
        q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
        v14_iqr = q75 - q25
        v14_cut_off = v14_iqr * 1.5
        v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off
        [x for x in v14_fraud if x < v14_lower or x > v14_upper]
        new_df = new_df.drop(
            new_df[(new_df['V14'] > v14_upper) | (new_df['V14'] < v14_lower)].index)

        # -----> V12 removing outliers from fraud transactions
        v12_fraud = new_df['V12'].loc[new_df[TARGET] == 1].values
        q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
        v12_iqr = q75 - q25
        v12_cut_off = v12_iqr * 1.5
        v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off
        [x for x in v12_fraud if x < v12_lower or x > v12_upper]
        new_df = new_df.drop(
            new_df[(new_df['V12'] > v12_upper) | (new_df['V12'] < v12_lower)].index)

        # Removing outliers V10 Feature
        v10_fraud = new_df['V10'].loc[new_df[TARGET] == 1].values
        q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
        v10_iqr = q75 - q25

        v10_cut_off = v10_iqr * 1.5
        v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off
        [x for x in v10_fraud if x < v10_lower or x > v10_upper]
        new_df = new_df.drop(
            new_df[(new_df['V10'] > v10_upper) | (new_df['V10'] < v10_lower)].index)

        # Undersampling before cross validating (prone to overfit)
        X = new_df.drop(TARGET, axis=1)
        y = new_df[TARGET]

        # Creating training, test and validation set
        X_train, X_rem, y_train, y_rem = train_test_split(
            X, y, test_size=0.3, random_state=SEED_SPLIT)
        X_test, X_val, y_test, y_val = train_test_split(
            X_rem, y_rem, test_size=0.33, random_state=SEED_SPLIT)

        # Turn the values into an array for feeding the classification algorithms.
        X_train = X_train.values
        X_test = X_test.values
        X_val = X_val.values
        y_train = y_train.values
        y_test = y_test.values
        y_val = y_val.values

        logger.info("train, test and val data was generated")
        return X_train, X_test, X_val, y_train, y_test, y_val
