import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sagemaker", action='store_true')
    parser.add_argument("--ratio", type=float, default=0.2)
    args = parser.parse_args()
    return args

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
    features: list
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

    return encoded


if __name__ == '__main__':

    args = get_args()
    SAGEMAKER=args.sagemaker
    RATIO=args.ratio
    container_base_path = "/opt/ml/processing"

    if SAGEMAKER:
        print('sagemaker')
        data_path = f"{container_base_path}/data/fitness_class_2212.csv"
        output_train_dir =  f"{container_base_path}/train"
        os.makedirs(output_train_dir, exist_ok=True)
        output_test_dir =  f"{container_base_path}/test"
        os.makedirs(output_test_dir, exist_ok=True)
    
    else:
        print('local')
        data_path = "data/fitness_class_2212.csv"
        output_train_dir = os.path.join(os.getcwd(), 'data/output/train')
        os.makedirs(output_train_dir, exist_ok=True)
        output_test_dir =  os.path.join(os.getcwd(), 'data/output/test')
        os.makedirs(output_test_dir, exist_ok=True)

    df = pd.read_csv(
        data_path,
    )
    transformed_df = transform(df, 'day_of_week', 'days_before')

    features = encode(
        transformed_df,
        'day_of_week', 
        'time', 
        [
            'months_as_member', 
            'weight', 
            'days_before', 
            'day_of_week', 
            'time', 
            'attended'
        ],  
    )
    X, y = features.drop('attended', axis=1), features['attended']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=RATIO, random_state=42
    )

    train = pd.concat(
        [X_train.reset_index(drop=True),y_train.reset_index(drop=True)],
        axis=1,
        ignore_index=True
    )
    test = pd.concat(
        [X_test.reset_index(drop=True),y_test.reset_index(drop=True)],
        axis=1,
        ignore_index=True
    )

    train.to_csv(f"{output_train_dir}/train.csv", index=False, header=False)
    test.to_csv(f"{output_test_dir}/test.csv", index=False, header=False)


