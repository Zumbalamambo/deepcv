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
cd model/classification
python eval_classifier.py --checkpoint_path=${CHECKPOINT_PATH} --dataset_dir=${DATASET_DIR} --dataset_name=cifar10 --model_name=mobilenet_v1


### YOLO Detection
python main.py --app=yolo -c config.ini config/yolo2/darknet-voc.cfg --task=detect --file=$FILE_PATH

