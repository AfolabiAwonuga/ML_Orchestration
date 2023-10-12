import pandas as pd
from typing import Type
from sklearn.base import BaseEstimator
from clearml import PipelineDecorator


@PipelineDecorator.component(
        return_values=['model'],
        # docker= 'folabinuga/fitness_package',
        repo='git@github.com:AfolabiAwonuga/ML_Orchestration.git',
        repo_branch='main'
)
def train_comp(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame
):
    from clearmll.fitness_package.train import xgb, svm, knn
    if model_name == 'xgb':
        model = xgb(X_train, y_train)

    elif model_name == 'svm':
        model = svm(X_train, y_train)

    else:
        model = knn(X_train, y_train)

    return model


@PipelineDecorator.component(
        return_values=['y_pred', 'accuracy', 'precision', 'recall', 'f1', 'matthews'],
        # docker= 'folabinuga/fitness_package',
        repo='git@github.com:AfolabiAwonuga/ML_Orchestration.git',
        repo_branch='main'
)
def eval_comp(
    model: Type[BaseEstimator],
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
):
    
    from clearmll.fitness_package.train import evaluate
    y_pred, accuracy, precision, recall, f1, matthews = evaluate(model, X_test, y_test)
    return y_pred, accuracy, precision, recall, f1, matthews


