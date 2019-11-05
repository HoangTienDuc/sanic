# Classifier


## Model

**MobileNet v2**

+ Paper: [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
+ Slides: [MobileNet v2 - Hiep Pham](https://docs.google.com/presentation/d/1gGl7ohpIU4bEmDCQoKJFVWnyDVF_nzw_ib4V4zJ8XbI/edit#slide=id.p)


## Train

+ Tutorials: [How to Retrain an Image Classifier for New Categories](https://www.tensorflow.org/hub/tutorials/image_retraining)

+ Transfer learning from ImageNet using Tensorflow Hub:

```
python retrain.py --image_dir /path/to/images-dir --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/1 --saved_model_dir /path/to/Classifier
```

*Example*: For dots classifier:

```
python retrain.py --image_dir dots --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/1 --saved_model_dir /home/hiepph/Classifier/dots/1
```
