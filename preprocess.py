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