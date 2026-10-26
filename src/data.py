import pandas as pd

def load_data(path="data/creditcard.csv"):
    df = pd.read_csv(path)
    return df

def time_aware_split(df, test_size=0.2):
    """ Делит данные по времени(прошлое в train, будущее в test) """
    df = df.sort_values('Time').reset_index(drop=True)

    split_idx = int(len(df) * (1 - test_size))

    split_time = df['Time'].iloc[split_idx]

    train_df = df[df['Time'] < split_time].copy()
    test_df = df[df['Time'] >= split_time].copy()

    print(f"Train shape: {train_df.shape} (Fraud: {train_df['Class'].sum()})")
    print(f"Test shape:  {test_df.shape} (Fraud: {test_df['Class'].sum()})")
    
    return train_df, test_df