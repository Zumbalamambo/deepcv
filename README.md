# deepcv
Can we make computer vision like our eyes?

## Prepare data
step1. download voc/coco data and uncompress them \
step2. modify the deepcv/config/dataset/voc.csv \
step3. python main.py -app=data -c config.cfg config/yolo2/darknet-voc.cfg


## Prepare Weights
https://pan.baidu.com/s/1nuKZnvF \
just download the log directory into deepcv, and uncompress it locally

## Training method:
### method 1: fine tuning net weights from pre-trained net
python main.py --app=yolo -c config.cfg config/yolo2/darknet-voc.cfg --task=train --transfer=$WEIGHTS_DRI
### method 2: training net weights from the beginning
python main.py --app=yolo -c config.cfg config/yolo2/darknet-voc.cfg --task=train
## Running method:

### classification
```shell
$ FILE_PATH = ~/dataset/
$ python main.py --config=config/vgg/vgg_16.cfg \ 
                 --app=classify \
                 --file=${FILE_PATH} or --file_url=${FILE_PATH}

```
# Pre-trained Models

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

### YOLO Detection
python main.py --app=yolo -c config.ini config/yolo2/darknet-voc.cfg --task=detect --file=$FILE_PATH

