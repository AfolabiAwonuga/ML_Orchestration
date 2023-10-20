import pandas as pd
import xgboost as xgb
from typing import Type
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def train_xgb(
        X_train: pd.DataFrame,
        y_train: pd.Series
) -> BaseEstimator:
    '''
    Trains an XGBoost classifier.
    
    Parameters:
        X_train (pd.DataFrame): The training data with features.
        y_train (pd.Series): The target labels for training data.

    Returns:
        BaseEstimator: The trained XGBoost classifier.
    '''
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {"objective": "binary:logistic", "eval_metric": "map"}
    xgb_model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, "train")],
        verbose_eval=0,
    )
    
    return xgb_model


def train_svm(
        X_train: pd.DataFrame,
        y_train: pd.Series
) -> BaseEstimator:
    '''
    Trains a Support Vector Machine (SVM) classifier.
    
    Parameters:
        X_train (pd.DataFrame): The training data with features.
        y_train (pd.Series): The target labels for training data.

    Returns:
        BaseEstimator: The trained SVM classifier.
    '''
    
    svm_model = SVC()
    svm_model.fit(X_train.values, y_train.values)

    return svm_model


def train_lr(
        X_train: pd.DataFrame,
        y_train: pd.Series
) -> BaseEstimator:
    '''
    Trains a Logistic Regression classifier.
    
    Parameters:
        X_train (pd.DataFrame): The training data with features.
        y_train (pd.Series): The target labels for training data.

    Returns:
        BaseEstimator: The trained Logistic Regression classifier.
    '''
    
    lr_model = LogisticRegression()
    lr_model.fit(X_train.values, y_train.values)

    return lr_model


def train_knn(
        X_train: pd.DataFrame,
        y_train: pd.Series
) -> BaseEstimator:
    '''
    Trains a k-Nearest Neighbors (k-NN) classifier.
    
    Parameters:
        X_train (pd.DataFrame): The training data with features.
        y_train (pd.Series): The target labels for training data.

    Returns:
        BaseEstimator: The trained k-NN classifier.
    '''
    
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train.values, y_train.values)

    return knn_model


def evaluate(
        model_name: str,
        model: Type[BaseEstimator],
        X_test: pd.DataFrame,
        y_test: pd.Series
) -> tuple:
    '''
    Evaluates a machine learning model using various metrics.
    
    Parameters:
        model (Type[BaseEstimator]): The trained machine learning model to be evaluated.
        X_test (pd.DataFrame): The test data with features.
        y_test (pd.Series): The true target labels for the test data.

    Returns:
        tuple: A tuple containing:
            y_pred (list): Predicted labels for the test data.
            accuracy (float): The accuracy of the model.
            precision (float): The precision of the model.
            recall (float): The recall of the model.
            f1 (float): The F1 score of the model.
            matthews (float): The Matthews correlation coefficient of the model.
    
    '''
    if model_name == 'xgb':
        dtest = xgb.DMatrix(X_test, label=y_test)
        preds = model.predict(dtest)

    else:
        preds = model.predict(X_test.values)
        
    y_pred = [1 if i > 0.5 else 0 for i in preds]
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    matthews = metrics.matthews_corrcoef(y_test, y_pred)

    return y_pred, accuracy, precision, recall, f1, matthews
    

    
    