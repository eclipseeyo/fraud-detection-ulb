import numpy as np
from data import load_data, time_aware_split
from features import prepare_features_and_scale
from evaluate import evaluate_model
from visualize import plot_pr_curves, plot_feature_importance
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_recall_curve, classification_report, average_precision_score


def find_best_treshold(y_true, y_probs):
    """ Ищем порог для максимального F1 """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    return best_threshold, best_f1


def main():
    df = load_data()

    train_df, test_df = time_aware_split(df, test_size=0.2)

    X_train, y_train, X_test, y_test = prepare_features_and_scale(train_df, test_df)

    lr = LogisticRegression(class_weight='balanced',
                            max_iter=1000,
                            random_state=42)
    lr.fit(X_train, y_train)
    y_probs_lr = lr.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, y_probs_lr, model_name="LogisticRegression")

    best_thr_lr, best_f1_lr = find_best_treshold(y_test, y_probs_lr)
    print(f"\n[LR] Best Threshold: {best_thr_lr:.3f}, Best F1: {best_f1_lr:.4f}")

    y_pred_lr_best = (y_probs_lr >= best_thr_lr).astype(int)
    print(classification_report(y_test, y_pred_lr_best, target_names=['Legit', 'Fraud']))


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
    y_probs_xgb = xgb.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, y_probs_xgb, model_name="XGBoost")

    best_thr_xgb, best_f1_xgb = find_best_treshold(y_test, y_probs_xgb)
    print(f"\n[XGB] Best Threshold: {best_thr_xgb:.3f}, Best F1: {best_f1_xgb:.4f}")
    y_pred_xgb_best = (y_probs_xgb >= best_thr_xgb).astype(int)
    print(classification_report(y_test, y_pred_xgb_best, target_names=['Legit', 'Fraud']))


    plot_pr_curves(y_test, y_probs_lr, y_probs_xgb,
                   average_precision_score(y_test, y_probs_lr),
                   average_precision_score(y_test, y_probs_xgb))

    plot_feature_importance(xgb, X_train.columns)


if __name__ == "__main__":
    main()