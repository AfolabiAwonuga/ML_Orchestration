import os 

os.system("pip install -U sagemaker")

import time
import boto3
import argparse
import sagemaker
import pandas as pd
from sklearn import metrics
from joblib import dump,load
from  sklearn.base import BaseEstimator
from sagemaker.experiments import Run, load_run
from sklearn.linear_model import LogisticRegression




def get_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--sagemaker', action='store_true')
    parser.add_argument('--train', type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument('--test', type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument('--train-file', type=str, default="train.csv")
    parser.add_argument('--test-file', type=str, default="test.csv")
    parser.add_argument('--model_dir', type=str, default=os.environ.get("SM_MODEL_DIR"))
    args = parser.parse_args()
    return args


def load_data(
        train_path: str,
        test_path: str
):
    train = pd.read_csv(
        train_path,
        header=None
    )
    test = pd.read_csv(
        test_path,
        header=None
    )

    return train, test


def train_lr(
        train: pd.DataFrame,
        dump_path: str
) -> BaseEstimator:
    '''
    Trains a Logistic Regression classifier.
    
    Parameters:
        X_train (pd.DataFrame): The training data with features.
        y_train (pd.Series): The target labels for training data.

    Returns:
        BaseEstimator: The trained Logistic Regression classifier.
    '''
    X = train.iloc[:, :-1]
    y = train.iloc[:, -1]

    lr_model = LogisticRegression()
    lr_model.fit(X.values, y.values)
    dump(lr_model, filename=dump_path, compress=9)


def load_model(model_dir):
    """
    Load the model for inference
    """
    loaded_model = load(model_dir)
    return loaded_model


def evaluate_model(
        model: BaseEstimator,
        test: pd.DataFrame,
        run
):
    X = test.iloc[:,:-1]
    y = test.iloc[:,-1]
    y_pred = model.predict(X)

    accuracy = metrics.accuracy_score(y, y_pred)
    precision = metrics.precision_score(y, y_pred)
    recall = metrics.recall_score(y, y_pred)
    f1 = metrics.f1_score(y, y_pred)

    run.log_confusion_matrix(y, y_pred)
    run.log_metric(name='test:accuracy', value=accuracy)
    run.log_metric(name='test:precision', value=precision)
    run.log_metric(name='test:recall', value=recall)
    run.log_metric(name='test:f1', value=f1)
    

    print(f'Acurracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')
    

def inference(input_data, model):
    """
    Apply model to the incoming request
    """
    return model.predict(input_data)
  

if __name__=="__main__":
    
    boto_session = boto3.session.Session(region_name='us-east-1')
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    s3 = boto3.client("s3")


    args = get_args()
    # SAGEMAKER = args.sagemaker
    TRAIN = args.train
    TEST = args.test
    TRAIN_FILE = args.train_file
    TEST_FILE = args.test_file
    MODEL_DIR = args.model_dir
    # EXPERIMENT_NAME = args.experiment_name experiment_name=EXPERIMENT_NAME, run_name=f'train-run{int(time.time())}'

    # if SAGEMAKER:
    print('sagemaker')
    train_data_path = os.path.join(TRAIN, TRAIN_FILE)
    test_data_path = os.path.join(TEST, TEST_FILE)

    train, test = load_data(train_data_path, test_data_path)
    train_lr(train, f"{MODEL_DIR}/lr_model.pkl")

    with load_run(sagemaker_session=sagemaker_session) as run:
        model = load_model(f"{MODEL_DIR}/lr_model.pkl")
        evaluate_model(model, test, run)

    # else:
    #     print("local")
    #     train_data_path = os.path.join(TRAIN, TRAIN_PATH)
    #     test_data_path = os.path.join(TEST, TEST_PATH)

    #     train, test = load_data(train_data_path, test_data_path)
    #     train_lr(train, f"{MODEL_DIR}/lr_model.pkl")

    #     model = load_model(f"{MODEL_DIR}/lr_model.pkl")
    #     evaluate_model(model, test)


# python training_script.py --train /Users/Labi/V_env/ML_Orchestration/sagemaker/data/output/train --test /Users/Labi/V_env/ML_Orchestration/sagemaker/data/output/test --train_path train.csv --test_path test.csv --model_dir /Users/Labi/V_env/ML_Orchestration/sagemaker/model

