import re
import os
import pandas as pd
from clearml import Dataset
from sklearn.model_selection import train_test_split


def get_data(
        dataset_name: str,
        pattern: str=None,
) -> str:
    '''
    Retrieves a local copy of a specified dataset from clearml.

    Parameters:
        dataset_name (str): The name of the dataset to retrieve.
        pattern (str, optional): A regular expression pattern to match a specific file within the dataset (default is None).

    Returns:
        str: The path to the retrieved data.
    '''
    
    ds_get = Dataset.get(
    dataset_name=dataset_name,
).get_local_copy()
    if pattern:
        data_path = os.path.join(ds_get, [element for element in os.listdir(ds_get) if re.match(pattern, element)][0])
    else:
        data_path = os.path.join(ds_get, os.listdir(ds_get)[0])

    return data_path


def upload_data(
        path, 
        name:str,
        project_name: str,
        storage: str,
        parent: str
) -> None:
    '''
    Uploads data to the specified clearml project.
    
    Parameters:
        path (str): The path to the data to be uploaded.
        name (str): The name of the dataset to be created.
        project_name (str): The name of the clearml project.
        storage (str): The URI for the storage location of the dataset.
        parent (str): The name of the parent dataset (if applicable).

    Returns:
        None
    '''
    
    ds = Dataset.create(
    dataset_name=name,
    dataset_project=project_name,
    parent_datasets=[parent],
    output_uri=storage
)
    ds.add_files(
    path=path
    )
    ds.finalize(auto_upload=True)


def transform(
        data: pd.DataFrame,
        col_1: str,
        col_2: str
) -> pd.DataFrame:
    '''
    Applies specific transformations to a DataFrame.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        col_1 (str): The name of the first column to be transformed.
        col_2 (str): The name of the second column to be transformed.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    '''
    
    data.dropna(inplace=True)
    transformed_data = data.copy()
    transformed_data[col_1] = transformed_data[col_1].apply(
        lambda x: 'Wed' if x == 'Wednesday' else ('Fri' if x == 'Fri.' else ('Mon' if x == 'Monday' else x))
        )
    transformed_data[col_2] = transformed_data[col_2].apply(lambda x: x.split(' ')[0])
    transformed_data[col_2] = transformed_data[col_2].astype('int')

    return transformed_data


def encode( 
        data: pd.DataFrame,
        col_1: str,
        col_2: str,
        features: list,
        kind: str
) -> str:
    '''
    Encodes specific columns and saves the resulting DataFrame to a CSV file.
    
    Parameters:
        data (pd.DataFrame): The input DataFrame.
        col_1 (str): The name of the first column to be encoded.
        col_2 (str): The name of the second column to be encoded.
        features (list): A list of features to include in the resulting DataFrame.
        kind (str): A descriptor for the kind of data being encoded.

    Returns:
        str: The path to the saved CSV file.
    '''

    ordinal_converter_day = lambda day: {
    'Mon': 1,
    'Tue': 2,
    'Wed': 3,
    'Thu': 4,
    'Fri': 5,
    'Sat': 6,
    'Sun': 7
    }.get(day)

    ordinal_converter_time = lambda time: {
        'AM': 1,
        'PM': 2,
    }.get(time) 

    data[col_1] = data[col_1].apply(ordinal_converter_day)
    data[col_2] = data[col_2].apply(ordinal_converter_time)
    encoded = data[features]
    dataset_path = f'data/{kind}/encoded_{kind}_data.csv'
    encoded.to_csv(dataset_path, index=False)

    return dataset_path


def split_data(
        data: pd.DataFrame,
        target: str
) -> tuple:
    '''
    Splits a DataFrame into features (X) and target labels (y) for training and testing.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing features and target labels.
        target (str): The name of the column containing the target labels.

    Returns:
        tuple: A tuple containing four DataFrames:
            X_train (pd.DataFrame): The training data features.
            X_test (pd.DataFrame): The testing data features.
            y_train (pd.Series): The training target labels.
            y_test (pd.Series): The testing target labels.
    '''
    
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test