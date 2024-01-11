import yolov5
import argparse
import json


class Inference:
    def __init__(self, model_path):
        # Load model
        self._model = yolov5.load(model_path)

    def execute(self, image_path, output_path):
        # Execute Inference
        results = self._model(image_path)

        # parse results
        predictions = results.pred[0]
        boxes = predictions[:, :4]  # x1, y1, x2, y2
        scores = predictions[:, 4]
        categories = predictions[:, 5]

        outputs = []
        for box, score, category in zip(boxes, scores, categories):
            x1, y1, x2, y2 = box
            outputs.append(
                {
                    "bbox": {
                        "left": float(x1),
                        "right": float(x2),
                        "top": float(y1),
                        "bottom": float(y2)
                    },
                    "confidence_score": float(score),
                    "category": int(category)
                }
            )

        with open(output_path, 'w') as outfile:
            json.dump(outputs, outfile, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_file_path', required=True)

    arguments = parser.parse_args()

    inference = Inference(
        arguments.model_path,
    )

    inference.execute(
        arguments.input_data_path,
        arguments.output_file_path
    )
