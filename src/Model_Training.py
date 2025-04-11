import mlflow
from torchvision.models.detection import fasterrcnn_resnet50_fpn

mlflow.start_run()
model = fasterrcnn_resnet50_fpn(pretrained=True)
mlflow.log_param("model", "Faster R-CNN")
mlflow.end_run()