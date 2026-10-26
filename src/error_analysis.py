import pandas as pd
import numpy as np
from data import load_data, time_aware_split
from features import prepare_features_and_scale
from xgboost import XGBClassifier

def main():
    df = load_data()
    train_df, test_df = time_aware_split(df, test_size=0.2)
    X_train, y_train, X_test, y_test = prepare_features_and_scale(train_df, test_df)
    
    neg_pos_ratio = (len(y_train) - y_train.sum()) / y_train.sum()
    
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=neg_pos_ratio,
        random_state=42,
        eval_metric='aucpr',
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    y_probs = xgb.predict_proba(X_test)[:, 1]
    
    threshold = 0.923
    y_pred = (y_probs >= threshold).astype(int)
    
    results = X_test.copy()
    results['True_Class'] = y_test
    results['Pred_Prob'] = y_probs
    results['Pred_Class'] = y_pred
    
    fp = results[(results['True_Class'] == 0) & (results['Pred_Class'] == 1)]
    fn = results[(results['True_Class'] == 1) & (results['Pred_Class'] == 0)]
    
    print(f"False Positives (False Alarms): {len(fp)}")
    if len(fp) > 0:
        print(f"  Mean Amount: {fp['Amount'].mean():.2f}")
        print(f"  Median Amount: {fp['Amount'].median():.2f}")
        print(f"  Mean Prob: {fp['Pred_Prob'].mean():.4f}\n")
    
    print(f"False Negatives (Missed Fraud): {len(fn)}")
    if len(fn) > 0:
        print(f"  Mean Amount: {fn['Amount'].mean():.2f}")
        print(f"  Median Amount: {fn['Amount'].median():.2f}")
        print(f"  Mean Prob: {fn['Pred_Prob'].mean():.4f}")
        print("\n  Probability distribution for missed fraud:")
        print(fn['Pred_Prob'].describe())

if __name__ == "__main__":
    main()