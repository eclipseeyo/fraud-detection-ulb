import numpy as np
import time
from data import load_data, time_aware_split
from features import prepare_features_and_scale
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, average_precision_score

def main():
    df = load_data()
    train_df, test_df = time_aware_split(df, test_size=0.2)
    X_train, y_train, X_test, y_test = prepare_features_and_scale(train_df, test_df)

    neg_pos_ratio = (len(y_train) - y_train.sum()) / y_train.sum()

    tscv = TimeSeriesSplit(n_splits=3)
    scorer = make_scorer(average_precision_score, needs_proba=True)

    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'scale_pos_weight': [neg_pos_ratio]
    }

    xgb = XGBClassifier(
        random_state=42, 
        eval_metric='aucpr', 
        n_jobs=-1
    )

    start_time = time.time()
    
    random_search = RandomizedSearchCV(
        xgb, 
        param_distributions=param_dist, 
        n_iter=20, 
        scoring=scorer, 
        cv=tscv, 
        verbose=2, 
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"\nSearch time: {time.time() - start_time:.1f} seconds.")
    
    print("\n=== BEST PARAMS ===")
    print(random_search.best_params_)
    print(f"Best CV PR-AUC: {random_search.best_score_:.4f}")

    best_model = random_search.best_estimator_
    y_probs_best = best_model.predict_proba(X_test)[:, 1]
    
    from evaluate import evaluate_model
    evaluate_model(y_test, y_probs_best, model_name="Tuned XGBoost")

if __name__ == "__main__":
    main()