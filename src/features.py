import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_features_and_scale(train_df, test_df):
    """ Добавляем Hour и масштабируем Time и Amount """

    train_df['Hour'] = (train_df['Time'] // 3600) % 24
    test_df['Hour'] = (test_df['Time'] // 3600) % 24

    cols_to_scale = ['Time', 'Amount']

    scaler = StandardScaler()

    train_df[cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])
    test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])

    X_train = train_df.drop(columns='Class')
    y_train = train_df['Class']

    X_test = test_df.drop(columns='Class')
    y_test = test_df['Class']

    return X_train, y_train, X_test, y_test