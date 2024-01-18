# UNet

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

## Prepare the model file
To run this sample code, you will need the UNet model file (.pth file).  
Please prepare the model file exported from the FastLabel app.

## Change the config file
[Here](./config/parameters.json) is the model's config file.  
 - image_size
   - This is the width and height of the image to be input into the model. The image will be resized to these dimensions before being input into the model.
 - device
   - This concerns the device used for inference. Please specify 'cpu' or use 'cuda' when employing GPT through CUDA.
 - class_number
   - This is the number of correct labels.
 - threshold
   - This is the threshold for approximating the output mask image. The value range is from 0 to 1, and pixels with a brightness greater than this value will be white, while those with a lower brightness will be black.

## How to execute the model
Please run the below command to execute the model
```bash
python scripts/inference.py \
--input_data_path <path_to_input_image> \
--model_path <path_to_model_file> \
--conf_file_path <path_to_config_file> \
--output_dir_path <path_to_output_directory>
```

The actual file path and the example used are as follows.
```bash
python scripts/inference.py \
--input_data_path "input/image.jpg" \
--model_path "models/best.pth" \
--conf_file_path "config/parameters.json" \
--output_dir_path "output"
```

## Output files
During model execution, two types of files will be output to the directory specified by the argument `output_dir_path`.
 - Output JSON file
   - In this case, information about the segmentation is output as a JSON file named output.json. Of particular note is the key 'segmentation/counts', which represents in bits where the mask image is filled in. Please refer to [this](https://qiita.com/harmegiddo/items/15d618e6a3620446512e).
 - Mask Images
   - For each class label, a mask image is output. If there are 4 class labels, 4 mask images will be output.