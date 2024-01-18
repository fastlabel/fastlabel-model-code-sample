import argparse
import json

import cv2
import numpy as np
import pycocotools.mask as mask_util
import segmentation_models_pytorch as smp
import torch
import torchvision
from PIL import Image


class Inference:

    def __init__(self, model_path: str, conf_file_path: str) -> None:

        # Load parameters
        with open(conf_file_path, 'r') as file:
            self._parameters = json.load(file)

        self._device = self._parameters["device"]
        self._resize_width = self._parameters["image_size"]["width"]
        self._resize_height = self._parameters["image_size"]["height"]
        self._threshold = self._parameters["threshold"]

        # Load model
        self._model = smp.Unet(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            in_channels=3,
            classes=self._parameters["class_number"] + 1,
        )
        state_dict = torch.load(
            model_path,
            map_location=torch.device(self._device),
        )
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()

        self._normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _resize_to_prediction_image(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        ar = height / width

        if height > width:
            width_dst = int(self._resize_height / ar)
            image = image.resize((width_dst, self._resize_height))
            x_bgn = (self._resize_width - width_dst) // 2
            y_bgn = 0
        else:
            height_dst = int(self._resize_width * ar)
            image = image.resize((self._resize_width, height_dst))
            x_bgn = 0
            y_bgn = (self._resize_height - height_dst) // 2
        image_padding = Image.new("RGB",
                                  (self._resize_width, self._resize_height),
                                  (0, 0, 0))
        image_padding.paste(image, (x_bgn, y_bgn))
        return image_padding

    def execute(self, image_path: str, output_path: str) -> None:

        image_np = cv2.imread(image_path)

        # Resize image
        (original_image_height, original_image_width, depth) = image_np.shape
        image_pillow = Image.fromarray(image_np)
        image_pillow = self._resize_to_prediction_image(image_pillow)
        image_np = np.array(image_pillow)
        (resize_image_height, resize_image_width, depth) = image_np.shape

        # Convert numpy image to Torch Tensor
        input_tensor = (
            torchvision.transforms.functional.to_tensor(image_np).view(
                1, 3, resize_image_height,
                resize_image_width).to(self._device))

        # Normalization
        input_tensor = self._normalize(input_tensor)

        # Predict
        output = torch.sigmoid(self._model(input_tensor))

        # Convert Torch Tensor to Numpy image
        output = output.to(self._device).detach().numpy()

        output_images = output.squeeze()
        output_json = []
        for idx, output_image_per_class in enumerate(output_images):
            if idx == 0:
                continue

            # Convert such that if the brightness of each pixel is above a threshold, it becomes white;
            # if it is below the threshold, it becomes black.
            predict_img = np.zeros([self._resize_height,
                                    self._resize_width]).astype(np.int8)
            predict_img = np.where(output_image_per_class >= self._threshold,
                                   255, predict_img)
            predict_img = np.where(output_image_per_class < self._threshold, 0,
                                   predict_img)

            # Resize back to the original dimensions of the image.
            img_save = Image.fromarray(predict_img)
            img_save = np.array(img_save.convert("L"))
            img_save = cv2.resize(
                img_save,
                dsize=(original_image_width, original_image_height),
                interpolation=cv2.INTER_NEAREST,
            )

            # Save image
            cv2.imwrite(f'{output_path}/class_{idx}.png', img_save)

            if len(np.unique(img_save).tolist()) == 1:
                continue

            # Convert mask image to pixel position info
            segmentation = mask_util.encode(
                np.array(img_save.tolist(), order="F", dtype="uint8"))

            output_json.append({
                "image_id": 0,
                "category_id": idx - 1,
                "bbox": [],
                "score": -1,
                "segmentation": {
                    "size": segmentation["size"],
                    "counts": segmentation["counts"].decode("utf-8"),
                },
            })

        with open(f'{output_path}/output.json', 'w') as outfile:
            json.dump(output_json, outfile, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--conf_file_path', required=True)
    parser.add_argument('--output_dir_path', required=True)

    arguments = parser.parse_args()

    inference = Inference(arguments.model_path, arguments.conf_file_path)

    inference.execute(arguments.input_data_path, arguments.output_dir_path)
