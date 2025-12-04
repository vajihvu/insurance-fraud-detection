from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix, precision_recall_curve, auc, average_precision_score
import matplotlib.pyplot as plt


def evaluate(model, X_test, y_test, name: str = "model"):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    roc = roc_auc_score(y_test, probs)
    ap = average_precision_score(y_test, probs)
    f1 = f1_score(y_test, preds)

    print(f"--- {name} ---")
    print("ROC AUC:", round(roc, 4))
    print("Average Precision (PR-AUC):", round(ap, 4))
    print("F1:", round(f1, 4))
    print(classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    plot_pr_curve(y_test, probs, title=f"PR curve ({name})")


def plot_pr_curve(y_true, y_scores, title: str = None):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title or f"PR curve - AUC={pr_auc:.4f}")
    plt.show()