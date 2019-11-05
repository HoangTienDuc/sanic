# Image cropper

Crop ID images using RetinaNet or, in case of failure, feature matching.

## Setup

Environment variable "FTI_IDCARD_HOME" has to be defined and project folder has to be recognized as Python module.

On Linux
```
export FTI_IDCARD_HOME=/path/to/project/
set PYTHONPATH=/path/to/project/
```

On Windows
```
set FTI_IDCARD_HOME=/path/to/project/
set PYTHONPATH=/path/to/project/
```

## Single call

Must be called from top project folder

```
python src/models/cropper/__init__.py --in_dir /input/directory/ --out_dir /output/directory/
```

Input and output directory defaults to /test/src/ and /test/result/ respectively.

## ImageCropper

```
class ImageCropper(save_file=CHECKCHECKPOINT_FILE_PATH, batch_size=100, use_cuda=True)
```
Arguments:
- save_file: path to checkpoint file, defaults to FTI_IDCARD_HOME/src/models/cropper/checkpoint/corners.pth
- batch_size
- use_cuda

### Public methods
- process_image(filename, out_dir): process a single image and save to out_dir
- process_images(filenames, out_dir): filenames is a list of strings, each a single image
