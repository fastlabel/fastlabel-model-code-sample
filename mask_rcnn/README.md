# MaskRCNN

## Python Version
The Python version used for operational testing is as follows. If the code doesn't work, try changing it to the version below.
```bash
$ python --version
Python 3.10.4
```

## Install Libraries
Please install the necessary Python libraries using the following command.
```bash
pip install -r requirements.txt
```
```bash
pip install git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13
```
## Prepare the model file
To run this sample code, you will need the MaskRCNN model file (.pth file).  
Please prepare the model file exported from the FastLabel app.

## Change the config file
[Here](./config/parameters.json) is the model's config file.
 - device
   - This concerns the device used for inference. Please specify 'cpu' or use 'cuda' when employing GPT through CUDA.
 - class_number
   - This is the number of correct labels.
 - model_config_file_path
   - Please specify the path to the config file of the model used for training. Choose the one you used for training from [this directory](https://github.com/facebookresearch/detectron2/tree/main/configs). Using `detectron2/configs/` from the Detectron2 repository as the root path, specify the path below it (for example, `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml`).

## How to execute the model
Please run the below command to execute the model
```bash
python scripts/inference.py \
--input_data_path <path_to_input_image> \
--model_path <path_to_model_file> \
--conf_file_path <path_to_config_file> \
--output_file_path <path_to_output_directory>
```

The actual file path and the example used are as follows.
```bash
python scripts/inference.py \
--input_data_path "input/image.jpg" \
--model_path "models/best.pth" \
--conf_file_path "config/parameters.json" \
--output_file_path "output/output.json"
```

## Output file
JSON file will be output to `output_file_path`.
Of particular note is the key `segmentation/counts`, which represents in bits where the mask image is filled in. Please refer to [this](https://qiita.com/harmegiddo/items/15d618e6a3620446512e).