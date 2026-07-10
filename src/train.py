from data import load_data, split_data
from evaluation import compare_models, get_best_model
from features import create_features
from models import pipelines
from visualization import plot_confusion_matrix, plot_precision_recall, plot_roc_curve, plot_variable_importance


# Load and preprocess the dataframe
df = load_data()
new_df = create_features(df)
X_train, X_test, y_train, y_test = split_data(new_df)

  
# Train the candidate models and compare evaluation metrics
results = compare_models(
    pipelines,
    X_train,
    X_test,
    y_train,
    y_test,
    threshold=0.3
)
print(results)


# Select best-performing model based on PR AUC
best_model_name, best_model = get_best_model(results)
best_pipeline = pipelines[best_model_name]

print(f"\nBest model: {best_model_name}")
print(f"PR AUC: {best_model['PR AUC']:.4f}")


# Visualize chosen model performance on test data
plot_confusion_matrix(best_pipeline, X_test, y_test, model_name=best_model_name)
plot_precision_recall(best_pipeline, X_test, y_test, model_name=best_model_name)
plot_roc_curve(best_pipeline, X_test, y_test, model_name=best_model_name)
plot_variable_importance(best_pipeline.named_steps['model'], X_train.columns, model_name=best_model_name)