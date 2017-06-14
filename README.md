# deepcv
Can we make computer vision like our eyes?

## Prepare data
step1. download voc/coco data and uncompress them \
step2. modify the deepcv/config/dataset/voc.tsv \
step3. python main.py -app=data -c config.ini config/yolo2/darknet-20.ini \

## Running method:

### classification
python main.py --app=mnist

### YOLO Detection
python main.py --app=demo_yolo --file=~/deepcv/cache/dataset/voc/test.tfrecord -c config.ini config/yolo2/darknet-20.ini

## Weights
https://pan.baidu.com/s/1nuKZnvF \
just download the log directory into deepcv, and uncompress it locally
