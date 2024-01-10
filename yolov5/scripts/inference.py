import yolov5


class Inference:
    def __init__(self, model_path):
        # Load model
        self._model = yolov5.load(model_path)

    def execute(self, image_path):
        # Execute Inference
        results = self._model(image_path)

        # parse results
        predictions = results.pred[0]
        boxes = predictions[:, :4]  # x1, y1, x2, y2
        scores = predictions[:, 4]
        categories = predictions[:, 5]

if
