from clearml import Dataset


def create_datasets(
        path: str,
        dataset_name: str,
        dataset_project: str,
        output_uri: str=None
) -> None:
    '''
    Creates a dataset and uploads files to the specified project on ClearML.

    Parameters:
        path (str): The path to the data to be uploaded.
        dataset_name (str): The name of the dataset to be created.
        dataset_project (str): The name of the project on ClearML.
        output_uri (str, optional): The URI for the storage location of the dataset (default is None).

    Returns:
        None
    '''
    ds = Dataset.create(
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        output_uri=output_uri
    )

    ds.add_files(
        path=path
        )

    ds.finalize(auto_upload=True)


if __name__ == '__main__':
    create_datasets(
        '/Users/Labi/V_env/ML_Orchestration/data/train',
        'fitness_class_train_data',
        'fitness_project'
        )
    
    create_datasets(
        '/Users/Labi/V_env/ML_Orchestration/data/inference',
        'fitness_class_inference_data',
        'fitness_project'
        )