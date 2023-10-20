import pandas as pd
from typing import Type
from sklearn.base import BaseEstimator
from clearml import PipelineDecorator


@PipelineDecorator.component(
        return_values=['model'],
        repo='git@github.com:AfolabiAwonuga/ML_Orchestration.git',
        repo_branch='main',
)
def train_comp(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> BaseEstimator:
    '''
    ClearML pipeline component implementing training a specified model.
    '''
    import xgboost as xgb
    from joblib import dump
    from clearml import Task
    from clearmll.fitness_package.train import train_xgb, train_svm, train_knn, train_lr
    task = Task.current_task()
    task.output_uri = 'https://files.clear.ml'
    if model_name == 'xgb':
        model = train_xgb(X_train, y_train)
        dump(model, filename="clearmll/models/xgb-model.pkl", compress=9)

    elif model_name == 'svm':
        model = train_svm(X_train, y_train)
        dump(model, filename="clearmll/models/svm-model.pkl", compress=9)
    
    elif model_name == 'lr':
        model = train_lr(X_train, y_train)
        dump(model, filename="clearmll/models/lr-model.pkl", compress=9)
        
    else:
        model = train_knn(X_train, y_train)
        dump(model, filename="clearmll/models/knn-model.pkl", compress=9)

    return model


@PipelineDecorator.component(
        return_values=['y_pred', 'accuracy', 'precision', 'recall', 'f1', 'matthews'],
        repo='git@github.com:AfolabiAwonuga/ML_Orchestration.git',
        repo_branch='main'
)
def eval_comp(
    model_name: str,
    model: Type[BaseEstimator],
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
) -> tuple:
    '''
    ClearML pipeline component evaluating model.
    '''

    from clearmll.fitness_package.train import evaluate
    y_pred, accuracy, precision, recall, f1, matthews = evaluate(model_name, model, X_test, y_test)
    return y_pred, accuracy, precision, recall, f1, matthews


