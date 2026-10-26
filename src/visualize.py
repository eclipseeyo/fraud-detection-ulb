import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
import pandas as pd

def plot_pr_curves(y_true, y_probs_lr, y_probs_xgb, pr_auc_lr, pr_auc_xgb):
    """Строим PR-curve для сравнения моделей"""
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))

    prec_lr, rec_lr, _ = precision_recall_curve(y_true, y_probs_lr)
    plt.plot(rec_lr, prec_lr, label=f'Logistic Regression (PR-AUC = {pr_auc_lr:.3f})', color='blue', linewidth=2)

    prec_xgb, rec_xgb, _ = precision_recall_curve(y_true, y_probs_xgb)
    plt.plot(rec_xgb, prec_xgb, label=f'XGBoost (PR-AUC = {pr_auc_xgb:.3f})', color='red', linewidth=2)
    
    baseline = y_true.sum() / len(y_true)
    plt.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline (Random) = {baseline:.4f}')
    
    plt.xlabel('Recall (Доля пойманного фрода)')
    plt.ylabel('Precision (Доля правильных срабатываний)')
    plt.title('Precision-Recall Curve: LR vs XGBoost')
    plt.legend()
    plt.tight_layout()
    plt.savefig('notebooks/pr_curve.png', dpi=300)
    plt.show()

def plot_feature_importance(model, feature_names, top_n=15):
    """Строим график важности признаков для XGBoost"""
    importances = model.feature_importances_
    
    # Создаем DataFrame для удобства
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
    plt.title(f'Top {top_n} Feature Importances (XGBoost)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('notebooks/feature_importance.png', dpi=300)
    plt.show()
    
    print(f"\nTop 5 most important features:")
    print(feat_imp.head(5))