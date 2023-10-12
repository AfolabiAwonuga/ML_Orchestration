import pandas as pd
from typing import Type
from sklearn import metrics
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def xgb(
        X_train: pd.DataFrame,
        y_train: pd.Series
):
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)
    return xgb_model


def svm(
        X_train: pd.DataFrame,
        y_train: pd.Series
):
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    return svm_model


def knn(
        X_train: pd.DataFrame,
        y_train: pd.Series
):
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
    return knn_model


def evaluate(
        model: Type[BaseEstimator],
        X_test: pd.DataFrame,
        y_test: pd.Series
):
    preds = model.predict(X_test)
    y_pred = [1 if i > 0.5 else 0 for i in preds]
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    matthews = metrics.matthews_corrcoef(y_test, y_pred)

    # Plot confussion matrix
#     plt.title("Confusion Matrix Logging")
#     ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=plt.gca())
#     plt.show()
    return y_pred, accuracy, precision, recall, f1, matthews
    

    
    