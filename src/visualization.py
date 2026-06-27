import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, precision_recall_curve


def plot_roc_curve(model, X_test, y_test):
    y_probs = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    
    plt.plot(fpr, tpr, color="yellowgreen")
    plt.title('Roc Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    
    
def plot_precision_recall(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    
    plt.plot(recall, precision, color="yellowgreen")
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()


def plot_confusion_matrix(model, X_test, y_test):
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=["Legitimate", "Fraud"],
        cmap="YlGn"
    )

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_variable_importance(model, feature_names):
    importances = model.feature_importances_
    
    plt.bar(range(len(importances)), importances, color="yellowgreen")
    plt.xticks(
        range(len(feature_names)),
        feature_names,
        rotation=90
    )
    
    plt.title('Feature importances')
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.show()