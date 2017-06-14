# deepcv
Can we make computer vision like our eyes?

## Prepare data
step1. download voc/coco data and uncompress them \
step2. modify the deepcv/config/dataset/voc.tsv \
step3. python main.py -app=data -c config.ini config/yolo2/darknet-20.ini \


## Prepare Weights
https://pan.baidu.com/s/1nuKZnvF \
just download the log directory into deepcv, and uncompress it locally

## Training method:
### method 1: fine tuning net weights from pre-trained net
python main.py --app=yolo -c config.cfg config/yolo2/darknet-voc.cfg --transfer=./cache/log/yolo2/darknet/voc
### method 2: training net weights from the beginning
python main.py --app=yolo -c config.cfg config/yolo2/darknet-voc.cfg
## Running method:

### classification
python main.py --app=mnist

### YOLO Detection
python main.py --app=yolo --file=~/deepcv/cache/dataset/voc/test.tfrecord -c config.ini config/yolo2/darknet-20.ini

