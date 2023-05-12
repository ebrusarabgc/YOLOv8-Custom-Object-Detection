from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
model.train(data="config.yaml", epochs=25, save_period=5)  # train the model

# Keys to add
# save_period:	Save checkpoint every x epochs (disabled if < 1)
# patience:	epochs to wait for no observable improvement for early stopping of training (default val is 50)

