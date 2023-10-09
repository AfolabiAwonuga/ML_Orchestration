import pandas as pd
from clearml import PipelineDecorator



@PipelineDecorator.component(
        return_values=['data_path'],
        repo='git@github.com:AfolabiAwonuga/ML_Orchestration.git',
        repo_branch='main'
)
def get_data_comp(
    name: str
):
    from preprocess import get_data
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
    import pandas as pd
    from preprocess import transform
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
    from preprocess import encode
    from sklearn import set_config
    set_config(transform_output='pandas')
    encoded_data = encode(data, 'day_of_week', 'time', 'category', 'booking_id')
    # dataset_path = '../data/encoded_data'
    # encoded_data.to_csv('../data/encoded_data', index=False)
    return encoded_data


@PipelineDecorator.component(
    return_values=['status'],
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
    from preprocess import upload_data
    upload_data(
        path,
        name,
        project_name, 
        storage,
        parent
    )

    status = 'Complete'
    return status
    