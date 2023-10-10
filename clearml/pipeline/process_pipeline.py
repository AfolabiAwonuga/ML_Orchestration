import os
from dotenv import load_dotenv
from clearml import PipelineDecorator, Task
from components import get_data_comp, tranform_comp, encode_comp, upload_dataset_comp
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
    name: str,
    project_name: str,
    storage: str,
    parent: str=None 
):
    step_1 = get_data_comp(name)
    step_2 = tranform_comp(step_1)
    step_3 = encode_comp(step_2)
    step_4 = upload_dataset_comp(step_3, name, project_name, storage, parent)

    # return accuracy


if __name__ == "__main__":
    # PipelineDecorator.set_default_execution_queue('default')
    PipelineDecorator.run_locally()
    pipeline(
         'fitness_class', 
         'fitness_project',
         's3://sagemaker-practice-bucket-nuga',
         'bd4e0541f4db4821b37d57a4ea0489ca'
        )