from config import CLASS_WEIGHT, MAX_DEPTH, MAX_ITER, N_ESTIMATORS, N_JOBS, RANDOM_STATE
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def logistic_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=MAX_ITER))
    ])
   
   
def logistic_smote_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('model', LogisticRegression(max_iter=MAX_ITER))
    ])
     

def logistic_under_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('undersampling', RandomUnderSampler(random_state=RANDOM_STATE)),
        ('model', LogisticRegression(max_iter=MAX_ITER))
    ])
    

def random_forest_pipeline():
    return Pipeline([
        ('model',
         RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            class_weight=CLASS_WEIGHT,
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE
        ))
    ])
    

def random_forest_smote_pipeline():
    return Pipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('model',
         RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            class_weight=CLASS_WEIGHT,
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE
        ))
    ])
         
       
def random_forest_under_pipeline():
    return Pipeline([
        ('undersampling', RandomUnderSampler(random_state=RANDOM_STATE)),
        ('model',
         RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            class_weight=CLASS_WEIGHT,
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE
        ))
    ])
           

def xgboost_pipeline():
    return Pipeline([
        ('model',
         XGBClassifier(
            scale_pos_weight=10,
            eval_metric='logloss',
            random_state=RANDOM_STATE
        ))
    ])  
    
           
pipelines = {
    'Logistic Regression': logistic_pipeline(),
    'Logistic Regression with Undersampling': logistic_under_pipeline(),
    'Logistic Regression with Oversampling': logistic_smote_pipeline(),
    'Random Forest': random_forest_pipeline(),
    'Random Forest with Undersampling': random_forest_under_pipeline(),
    'Random Forest with Oversampling': random_forest_smote_pipeline(),
    'XGBoost': xgboost_pipeline()
}