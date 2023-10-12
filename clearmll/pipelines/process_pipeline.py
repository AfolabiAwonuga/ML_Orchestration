import os
from dotenv import load_dotenv
from clearml import PipelineDecorator, Task
from clearmll.components.process_components import get_data_comp, tranform_comp, encode_comp, upload_dataset_comp
load_dotenv()

CLEARML_WEB_HOST = os.getenv("CLEARML_WEB_HOST")
CLEARML_API_HOST = os.getenv("CLEARML_API_HOST")
CLEARML_FILES_HOST = os.getenv("CLEARML_FILES_HOST")
CLEARML_API_ACCESS_KEY = os.getenv("CLEARML_API_ACCESS_KEY")
CLEARML_API_SECRET_KEY = os.getenv("CLEARML_API_SECRET_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")


@PipelineDecorator.pipeline(name="transform pipeline", project="fitness_project")
def pipeline(
    dataset_name: str,
    project_name: str,
    storage: str,
    parent: str=None 
):
    print("launch step one -------------> data_path")
    data_path = get_data_comp(dataset_name)

    print("launch step two -------------> transformed_data")
    transformed_data = tranform_comp(data_path)

    print("launch step three -------------> encoded_data")
    encoded_data = encode_comp(transformed_data)

    print("launch step four -------------> encoded_data_upload")
    encoded_data_upload = upload_dataset_comp(encoded_data, dataset_name, project_name, storage, parent)


if __name__ == "__main__":
    # PipelineDecorator.set_default_execution_queue('default')
    PipelineDecorator.run_locally()
    pipeline(
         'fitness_class', 
         'fitness_project',
         's3://sagemaker-practice-bucket-nuga',
         'bd4e0541f4db4821b37d57a4ea0489ca'
        )