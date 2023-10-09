import pandas as pd
from clearml import PipelineDecorator
from sklearn import set_config
set_config(transform_output='pandas')

import os
import pandas as pd
from clearml import Dataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


def get_data(
        name: str

):
    ds_get = Dataset.get(
    dataset_name=name,
).get_local_copy()
    data_path = os.path.join(ds_get, os.listdir(ds_get)[0])

    return data_path


def upload_data(
        path, 
        name:str,
        project_name: str,
        storage: str,
        parent: str=None
):
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
    transformed_data = data.copy()
    transformed_data['day_of_week'] = transformed_data['day_of_week'].apply(
        lambda x: 'Wed' if x == 'Wednesday' else ('Fri' if x == 'Fri.' else x)
        )
    transformed_data[col_1] = transformed_data[col_1].apply(lambda x: x.split(' ')[0])
    transformed_data[col_2] = transformed_data[col_2].astype('int')

    return transformed_data


def encode( 
        data: pd.DataFrame,
        col_1: str,
        col_2: str,
        col_3: str,
        drop_col: str
) -> pd.DataFrame:

    day_enc = OrdinalEncoder()
    time_enc = OrdinalEncoder()
    col_transform = ColumnTransformer([
        ('Ordinal encoding day', day_enc, [col_1]),
        ('Ordinal encoding time', time_enc, [col_2]),
        ('OneHot', OneHotEncoder(sparse_output=False), [col_3])
    ], 
    remainder='passthrough',
    verbose_feature_names_out=False
    )
    encodeed = col_transform.fit_transform(data).drop(drop_col, axis = 1)

    return encodeed


@PipelineDecorator.component(
        return_values=['data_path'],
        repo='git@github.com:AfolabiAwonuga/ML_Orchestration.git',
        repo_branch='main'
)
def get_data_comp(
    name: str
):
    # from preprocess import get_data
    data_path = get_data(name)
    return data_path


@PipelineDecorator.component(
    return_values=['transformed_data'],
    cache=True,
    repo='git@github.com:AfolabiAwonuga/ML_Orchestration.git',
    repo_branch='main'
)
def tranform_comp(
    data_path: str
):
    # from preprocess import transform
    data = pd.read_csv(data_path)
    transformed_data = transform(data, 'day_of_week', 'days_before')
    return transformed_data


@PipelineDecorator.component(
    return_values=['dataset_path'],
    cache=True,
    repo='git@github.com:AfolabiAwonuga/ML_Orchestration.git',
    repo_branch='main'
)
def encode_comp(
    data: pd.DataFrame
):
    # from preprocess import encode
    encoded_data = encode(data, 'day_of_week', 'time', 'category', 'booking_id')
    dataset_path = '../data/encoded_data'
    encoded_data.to_csv('../data/encoded_data', index=False)
    return dataset_path


@PipelineDecorator.component(
    return_values=[None],
    cache=True,
    repo='git@github.com:AfolabiAwonuga/ML_Orchestration.git',
    repo_branch='main'
)
def upload_dataset_comp(
    path: str,
    name: str,
    project_name: str,
    storage: str,
    parent: str=None

):
    # from preprocess import upload_data
    upload_data(
        path,
        name,
        project_name, 
        storage,
        parent
    )
    