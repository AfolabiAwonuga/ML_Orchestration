from clearml import PipelineDecorator
from clearmll.components.process_components import get_data_comp, tranform_comp, encode_comp, upload_dataset_comp


@PipelineDecorator.pipeline(name="transform pipeline", project="fitness_project")
def pipeline(
    dataset_name: str,
    project_name: str,
    kind: str,
    storage: str=None,
    parent: str=None,
) -> None:
    '''
    ClearML transform pipeline.
    '''
    
    data_path = get_data_comp(dataset_name)
    transformed_data = tranform_comp(data_path)
    encoded_data = encode_comp(transformed_data, kind)
    encoded_data_upload = upload_dataset_comp(encoded_data, dataset_name, project_name, storage, parent)

if __name__ == "__main__":
    PipelineDecorator.run_locally()
    pipeline(
         'fitness_class_inference_data', # fitness_class_train_data, fitness_class_inference_data
         'fitness_project',
         kind ='inference',  # train, inference
         parent='4a6fc89c3f47470ebb5433c533197a33' # 1d019b1562fa42d0b4b23563af6a6ecc, 4a6fc89c3f47470ebb5433c533197a33
        )