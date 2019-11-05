## Create tfrecord

```
cd data
python create_individual_tfrecord.py --images-dir $DATA_DIR/jpg --annotation-file $DATA_DIR/train.txt --output $DATA_DIR/train.tfrecord --class_name name
```

```
python create_all_tfrecord.py --data_dir $DATA_DIR --phase train --output $DATA_DIR/train.tfrecord
```

## Train

+ Get inception resnet v2 weights:

```
wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz --directory-prefix weights
```

+ Clone tensorflow slim:

```
git clone https://github.com/tensorflow/models $HOME/src/tensorflow/models
export PYTHONPATH="$PYTHONPATH:$HOME/src/tensorflow/models/research/slim"
```

+ Train:

```
python train.py --dataset_dir $DATA_DIR --class_name all --optimizer adadelta --learning_rate 1.0 --save_interval_secs 90 --checkpoint_inception weights/inception_resnet_v2_2016_08_30.ckpt --train_log_dir ~/attention_ocr/train/name
```

+ Train with finetune:

```
python train.py --dataset_dir $DATA_DIR --class_name all --optimizer adadelta --learning_rate 0.5 --save_interval_secs 90 --checkpoint ~/attention_ocr/train/all/model.ckpt-225708 --train_log_dir ~/attention_ocr/train/name --freeze_cnn
```

## Eval

```
python eval.py --dataset_dir $DATA_DIR --class_name all --split_name test --train_log_dir ~/attention_ocr/train/back/features-new/ --eval_log_dir ~/attention_ocr/eval/back/features-new
```

## Export

```
python export.py --dataset_dir $DATA_DIR --class_name name --checkpoint ./checkpoints/new_train_inception_v3/model.ckpt-8870 --saved_dir $EXPORT_DIR/name/1
```

## Config

+ `READER_TRANSFORM`: `pad` resizes image to height and keeps ratio by padding width, `resize` breaks ratio


# Benchmark

```
cd benchmark
python benchmark.py --data_dir $DATA_DIR/old --label_file $DATA_DIR/old.csv --service http://10.3.9.223:5005
```
