# deepcv
Can we make computer vision like our eyes?

## License
> Copyright 2017 *** Authors. All Rights Reserved.

> Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.

> You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

> See the License for the specific language governing permissions and
  limitations under the License.



## Prepare data
```shell
$ python main.py --config=config/dataset/pascal_voc.cfg
                 --app=dataset

```

## classification
### classify
```shell
$ FILE_PATH = ~/dataset/test.jpg
$ python main.py --config=config/vgg/vgg_16.cfg
                 --gpu = True
                 --app=classifier
                 --task=classify
                 --file=${FILE_PATH} or --file_url=${FILE_PATH}
```
### train a classification model from scratch
```shell
DATASET_DIR=cache/dataset/imagenet
LOG_DIR=cache/log/vgg_16
python train_classifier.py --model_name=vgg_16
                           --log_dir=${LOG_DIR}
                           --dataset_dir=${DATASET_DIR}
                           --dataset_name=imagenet
                           --dataset_split_name=train
```

### fine-tuning a classification model from an existing checkpoint
```shell
$ DATASET_DIR=cache/dataset/imagenet
$ TRAIN_DIR=cache/log/vgg_16
$ CHECKPOINT_PATH=cache/weight/vgg_16.ckpt
$ python train_classifier.py --model_name=vgg_16
                             --train_dir=${TRAIN_DIR}
                             --dataset_dir=${DATASET_DIR}
                             --dataset_name=imagenet
                             --dataset_split_name=train
                             --checkpoint_path=${CHECKPOINT_PATH}
```
### YOLO Detection
```shell
$ FILE_PATH = ~/dataset/test.jpg
$ python main.py  --config=config/ssd/ssd_v1.cfg \
                  --app=detector \
                  --task=detect \
                  --file=$FILE_PATH
```

## Pre-trained Models

Model | TF-Slim File | Checkpoint | Top-1 Accuracy| Top-5 Accuracy |
:----:|:------------:|:----------:|:-------:|:--------:|
[Inception V1](http://arxiv.org/abs/1409.4842v1)|[Code](https://github.com/xhzcyc/deepcv/blob/master/model/classification/inception_v1.py)|[inception_v1_2016_08_28.tar.gz](https://pan.baidu.com/s/1gfACLMV)|69.8|89.6|
[Inception V2](http://arxiv.org/abs/1502.03167)|[Code](https://github.com/xhzcyc/deepcv/blob/master/model/classification/inception_v2.py)|[inception_v2_2016_08_28.tar.gz](https://pan.baidu.com/s/1gfACLMV)|73.9|91.8|
[Inception V3](http://arxiv.org/abs/1512.00567)|[Code](https://github.com/xhzcyc/deepcv/blob/master/model/classification/inception_v3.py)|[inception_v3_2016_08_28.tar.gz](https://pan.baidu.com/s/1gfACLMV)|78.0|93.9|
[Inception V4](http://arxiv.org/abs/1602.07261)|[Code](https://github.com/xhzcyc/deepcv/blob/master/model/classification/inception_v4.py)|[inception_v4_2016_09_09.tar.gz](https://pan.baidu.com/s/1gfACLMV)|80.2|95.2|
[Inception-ResNet-v2](http://arxiv.org/abs/1602.07261)|[Code](https://github.com/xhzcyc/deepcv/blob/master/model/classification/inception_resnet_v2.py)|[inception_resnet_v2_2016_08_30.tar.gz](https://pan.baidu.com/s/1gfACLMV)|80.4|95.3|
[ResNet 50](https://arxiv.org/abs/1512.03385)|[Code](https://github.com/xhzcyc/deepcv/blob/master/model/classification/resnet_v1.py)|[resnet_v1_50_2016_08_28.tar.gz](https://pan.baidu.com/s/1gfACLMV)|75.2|92.2|
[ResNet 101](https://arxiv.org/abs/1512.03385)|[Code](https://github.com/xhzcyc/deepcv/blob/master/model/classification/resnet_v1.py)|[resnet_v1_101_2016_08_28.tar.gz](https://pan.baidu.com/s/1gfACLMV)|76.4|92.9|
[ResNet 152](https://arxiv.org/abs/1512.03385)|[Code](https://github.com/xhzcyc/deepcv/blob/master/model/classification/resnet_v1.py)|[resnet_v1_152_2016_08_28.tar.gz](https://pan.baidu.com/s/1gfACLMV)|76.8|93.2|
[ResNet V2 200](https://arxiv.org/abs/1603.05027)|[Code](https://github.com/xhzcyc/deepcv/blob/master/model/classification/resnet_v2.py)|[TBA]()|79.9\*|95.2\*|
[VGG 16](http://arxiv.org/abs/1409.1556.pdf)|[Code](https://github.com/xhzcyc/deepcv/blob/master/model/classification/vgg.py)|[vgg_16_2016_08_28.tar.gz](https://pan.baidu.com/s/1gfACLMV)|71.5|89.8|
[VGG 19](http://arxiv.org/abs/1409.1556.pdf)|[Code](https://github.com/xhzcyc/deepcv/blob/master/model/classification/vgg.py)|[vgg_19_2016_08_28.tar.gz](https://pan.baidu.com/s/1gfACLMV)|71.1|89.8|

Choose the right MobileNet model to fit your latency and size budget. The size of the network in memory and on disk is proportional to the number of parameters. The latency and power usage of the network scales with the number of Multiply-Accumulates (MACs) which measures the number of fused Multiplication and Addition operations. These MobileNet models have been trained on the
[ILSVRC-2012-CLS](http://www.image-net.org/challenges/LSVRC/2012/)
image classification dataset. Accuracies were computed by evaluating using a single image crop.

Model Checkpoint | Million MACs | Million Parameters | Top-1 Accuracy| Top-5 Accuracy |
:----:|:------------:|:----------:|:-------:|:-------:|
[MobileNet_v1_1.0_224](https://pan.baidu.com/s/1gfACLMV)|569|4.24|70.7|89.5|
[MobileNet_v1_1.0_192](https://pan.baidu.com/s/1gfACLMV)|418|4.24|69.3|88.9|
[MobileNet_v1_1.0_160](https://pan.baidu.com/s/1gfACLMV)|291|4.24|67.2|87.5|
[MobileNet_v1_1.0_128](https://pan.baidu.com/s/1gfACLMV)|186|4.24|64.1|85.3|
[MobileNet_v1_0.75_224](https://pan.baidu.com/s/1gfACLMV)|317|2.59|68.4|88.2|
[MobileNet_v1_0.75_192](https://pan.baidu.com/s/1gfACLMV)|233|2.59|67.4|87.3|
[MobileNet_v1_0.75_160](https://pan.baidu.com/s/1gfACLMV)|162|2.59|65.2|86.1|
[MobileNet_v1_0.75_128](https://pan.baidu.com/s/1gfACLMV)|104|2.59|61.8|83.6|
[MobileNet_v1_0.50_224](https://pan.baidu.com/s/1gfACLMV)|150|1.34|64.0|85.4|
[MobileNet_v1_0.50_192](https://pan.baidu.com/s/1gfACLMV)|110|1.34|62.1|84.0|
[MobileNet_v1_0.50_160](https://pan.baidu.com/s/1gfACLMV)|77|1.34|59.9|82.5|
[MobileNet_v1_0.50_128](https://pan.baidu.com/s/1gfACLMV)|49|1.34|56.2|79.6|
[MobileNet_v1_0.25_224](https://pan.baidu.com/s/1gfACLMV)|41|0.47|50.6|75.0|
[MobileNet_v1_0.25_192](https://pan.baidu.com/s/1gfACLMV)|34|0.47|49.0|73.6|
[MobileNet_v1_0.25_160](https://pan.baidu.com/s/1gfACLMV)|21|0.47|46.0|70.7|
[MobileNet_v1_0.25_128](https://pan.baidu.com/s/1gfACLMV)|14|0.47|41.3|66.2|



