import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


def evaluate_model(model, X_test, y_test, threshold=0.3):
    """Compute evaluation metrics for a trained model."""
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    
    return {
        'F1': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'PR AUC': average_precision_score(y_test, y_prob),
        'ROC AUC': roc_auc_score(y_test, y_prob)
    }
   
    
def compare_models(models, X_train, X_test, y_train, y_test, threshold=0.3):
    """ 
    Fits each model on the training data and evaluates its performance on
    the test data using 'evaluate_model'.
    
    Returns:
    pd.DataFrame: The evaluation metrics for all models as a DataFrame.
    """
        
    results = []
    
    for name, model in models.items():      
        model.fit(X_train, y_train)
        
        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )
        
        metrics['Model'] = name
        
        results.append(metrics)
    return pd.DataFrame(results)

def get_best_model(results, metric='PR AUC'):
    """Return the best-performing model and its evaluation metrics."""
    best_row = results.sort_values(metric, ascending=False).iloc[0]
    return best_row['Model'], best_row