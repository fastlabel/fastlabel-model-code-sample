import argparse
import json

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import BoxMode, Instances


class Inference:

    def __init__(self, model_path: str, conf_file_path: str) -> None:

        # Load parameters
        with open(conf_file_path, 'r') as file:
            self._parameters = json.load(file)

        self._device = self._parameters["device"]
        self._class_number = self._parameters["class_number"]

        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                self._parameters["model_config_file_path"]))
        cfg.MODEL.DEVICE = self._device
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(self._class_number)
        cfg.MODEL.WEIGHTS = model_path

        self._model = DefaultPredictor(cfg)

    @staticmethod
    def _instances_to_coco_json(instances: Instances) -> list[dict]:
        """
        Ref: https://detectron2.readthedocs.io/en/latest/_modules/detectron2/evaluation/coco_evaluation.html
        Dump an "Instances" object to a COCO-format json that's used for evaluation.

        Args:
            instances (Instances):

        Returns:
            list[dict]: list of json annotations in COCO format.
        """
        num_instance = len(instances)
        if num_instance == 0:
            return []

        boxes = instances.pred_boxes.tensor.numpy()
        boxes = boxes.tolist()
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()

        has_keypoints = instances.has("pred_keypoints")
        if has_keypoints:
            keypoints = instances.pred_keypoints

        results: list[dict] = []
        for k in range(num_instance):
            result = {
                "image_id": 0,  # Not used
                "category_id": classes[k],
                "bbox": boxes[k],
                "score": scores[k],
            }
            if has_keypoints:
                # In COCO annotations,
                # keypoints coordinates are pixel indices.
                # However our predictions are floating point coordinates.
                # Therefore we subtract 0.5 to be consistent with the annotation format.
                # This is the inverse of data loading logic in `datasets/coco.py`.
                keypoints[k][:, :2] -= 0.5
                result["keypoints"] = keypoints[k].flatten().tolist()
            results.append(result)
        return results

    def execute(self, image_path: str, output_path: str) -> None:

        image = cv2.imread(image_path)
        results = self._model(image)
        output_json = self._instances_to_coco_json(results["instances"])

        with open(output_path, 'w') as outfile:
            json.dump(output_json, outfile, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--conf_file_path', required=True)
    parser.add_argument('--output_file_path', required=True)

    arguments = parser.parse_args()

    inference = Inference(arguments.model_path, arguments.conf_file_path)

    inference.execute(arguments.input_data_path, arguments.output_file_path)
