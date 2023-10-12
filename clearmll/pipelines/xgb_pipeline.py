import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from clearml import PipelineDecorator, Logger
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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


@PipelineDecorator.pipeline(name="train pipeline", project="fitness_project")
def train_pipeline(
    dataset_name: str,
    target: str,
    model_name: str,
):
    print("launch step one -------------> data_path")
    data_path = get_data_comp(dataset_name)
    
    print("launch step two -------------> split_data")
    X_train, X_test, y_train, y_test = split_dataset_comp(data_path, target)

    print("launch step three -------------> trained_model")
    trained_model = train_comp(model_name, X_train, y_train)

    print("launch step four -------------> evaluate")
    y_pred, accuracy, precision, recall, f1, matthews = eval_comp(trained_model, X_test, y_test)

    plt.title("Confusion Matrix Logging")
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=plt.gca())
    plt.show()

    # print(f"Accuracy={accuracy}%") 
    # print(f"precision={precision}%") 
    # print(f"recall={recall}%") 
    # print(f"f1={f1}%")
    # print(f"matthews={matthews}%") 
    Logger.current_logger().report_scalar(title='Accuracy', series='series', iteration=0, value=accuracy)
    Logger.current_logger().report_scalar(title='Precision', series='series', iteration=0, value=precision)
    Logger.current_logger().report_scalar(title='Recall', series='series', iteration=0, value=recall)
    Logger.current_logger().report_scalar(title='F1', series='series', iteration=0, value=f1)
    Logger.current_logger().report_scalar(title='Matthews', series='series', iteration=0, value=matthews)


if __name__ == "__main__":
    # PipelineDecorator.set_default_execution_queue('default')
    PipelineDecorator.run_locally()
    train_pipeline(
         'fitness_class', 
         'attended',
         'xgb',
        )