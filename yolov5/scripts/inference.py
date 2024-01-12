import argparse
import json

import yolov5


class Inference:

    def __init__(self, model_path: str, conf_file_path: str) -> None:
        # Load model
        self._model = yolov5.load(model_path)

        # Load parameters
        with open(conf_file_path, 'r') as file:
            self._parameters = json.load(file)

        # Set model parameters
        self._model.conf = self._parameters["confidence_threshold"]
        self._model.iou = self._parameters["iou_threshold"]
        self._model.agnostic = self._parameters["class_agnostic"]
        self._model.multi_label = self._parameters["multiple_label"]
        self._model.max_det = self._parameters["max_num_detection"]

    def execute(self, image_path: str, output_path: str) -> None:
        # Execute Inference
        results = self._model(image_path)

        # Parse results
        predictions = results.pred[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        categories = predictions[:, 5]

        outputs = []
        for box, score, category in zip(boxes, scores, categories):
            x1, y1, x2, y2 = box
            outputs.append({
                "bbox": {
                    "left": float(x1),
                    "right": float(x2),
                    "top": float(y1),
                    "bottom": float(y2)
                },
                "confidence_score":
                float(score),
                "category":
                self._parameters["class_name_and_index"][str(int(category))]
            })

        with open(output_path, 'w') as outfile:
            json.dump(outputs, outfile, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--conf_file_path', required=True)
    parser.add_argument('--output_file_path', required=True)

    arguments = parser.parse_args()

    inference = Inference(arguments.model_path, arguments.conf_file_path)

    inference.execute(arguments.input_data_path, arguments.output_file_path)
