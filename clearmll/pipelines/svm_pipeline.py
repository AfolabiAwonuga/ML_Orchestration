import os
from dotenv import load_dotenv
from clearml import PipelineDecorator, Task
from clearmll.components.train_components import train_comp, eval_comp 
from clearmll.components.process_components import get_data_comp, split_dataset_comp
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
def train_pipeline(
    dataset_name: str,
    target: str,
    model_name: str,
):
    get_data_path = get_data_comp(dataset_name)
    X_train, X_test, y_train, y_test = split_dataset_comp(get_data_path, target)
    trained_model = train_comp(model_name, X_train, y_train)
    evaluate = eval_comp(trained_model, X_test, y_test)


if __name__ == "__main__":
    # PipelineDecorator.set_default_execution_queue('default')
    PipelineDecorator.run_locally()
    train_pipeline(
         'fitness_class', 
         'attended',
         'svm',
        )