from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score, classification_report

def evaluate_model(y_true, y_probs, model_name="Model"):
    """ Считаем PR-AUC и F1 """

    pr_auc = average_precision_score(y_true, y_probs)

    threshold = 0.5
    y_pred = (y_probs >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred)

    print(f"=== {model_name} ===")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"F1 (threshold={threshold}): {f1:.4f}")
    print(f"\nClassification Report (threshold={threshold}):")
    print(classification_report(y_true, y_pred, target_names=['Legit', 'Fraud']))
    
    return pr_auc, f1