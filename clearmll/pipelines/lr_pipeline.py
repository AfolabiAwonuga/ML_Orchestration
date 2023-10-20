import matplotlib.pyplot as plt
from clearml import PipelineDecorator, Task
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from clearmll.components.train_components import train_comp, eval_comp 
from clearmll.components.process_components import get_data_comp, split_dataset_comp



@PipelineDecorator.pipeline(name="train pipeline", project="fitness_project")
def train_pipeline(
    dataset_name: str,
    pattern: str,
    target: str,
    model_name: str,
):
    '''
    ClearML pipeline trains Logistic regression classifier.
    '''

    data_path = get_data_comp(dataset_name, pattern)
    X_train, X_test, y_train, y_test = split_dataset_comp(data_path, target)
    trained_model = train_comp(model_name, X_train, y_train)
    y_pred, accuracy, precision, recall, f1, matthews = eval_comp(trained_model, X_test, y_test)

    Task.current_task().get_logger().report_single_value(name="Accuracy", value=accuracy)
    Task.current_task().get_logger().report_single_value(name="Precision", value=precision)
    Task.current_task().get_logger().report_single_value(name="Recall", value=recall)
    Task.current_task().get_logger().report_single_value(name="F1", value=f1)
    Task.current_task().get_logger().report_single_value(name="Matthews", value=matthews)

    plt.title("Confusion Matrix Logging")
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=plt.gca())
    plt.show()


if __name__ == "__main__":
    PipelineDecorator.run_locally()
    train_pipeline(
         'fitness_class_train_data', 
         r'^encoded.*',
         'attended',
         'lr',
        )