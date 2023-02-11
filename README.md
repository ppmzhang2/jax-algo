# Jax-Algorithms

Computer vision deep learning algorithms implemented with
[JAX](https://jax.readthedocs.io/en/latest/) and
[Haiku](https://dm-haiku.readthedocs.io/en/latest/).

## Deploy

```sh
# CPU-only running environment deploy
make deploy-cpu
# CUDA running environment
make deploy-gpu
# dev environment
make deploy-dev
```

## MNIST Classification with ResNet

start training with the MNIST handwritten digit:

```sh
pdm run resnet train-resnet --n-epoch=100
```

## Object Detection with YOLOv3

Preparing batabase:

```sh
# reset training DB
pdm run yolov3 db-reset --train=True
# reset test DB
pdm run yolov3 db-reset --train=False
```

Create COCO CSV annotation files:

```sh
# training CSVs
pdm run yolov3 coco-annot-to-csv \
    --in-json="data/coco_instances_train2014.json" \
    --img-folder="data/coco_train2014" \
    --imgtag-csv="data/coco_imgtag_train2014.csv" \
    --cate-csv="data/coco_cate_train2014.csv" \
    --box-csv="data/coco_box_train2014.csv"
# validation CSVs
pdm run yolov3 coco-annot-to-csv \
    --in-json="data/coco_instances_val2014.json" \
    --img-folder="data/coco_val2014" \
    --imgtag-csv="data/coco_imgtag_val2014.csv" \
    --cate-csv="data/coco_cate_val2014.csv" \
    --box-csv="data/coco_box_val2014.csv"
```

Load into sqlite annotation CSV files:

```sh
# load training annotation
pdm run yolov3 load-coco-annot \
    --imgtag-csv="./data/coco_imgtag_train2014.csv" \
    --cate-csv="./data/coco_cate_train2014.csv" \
    --box-csv="./data/coco_box_train2014.csv" \
    --train=True
# load validation annotation
pdm run yolov3 load-coco-annot \
    --imgtag-csv="./data/coco_imgtag_val2014.csv" \
    --cate-csv="./data/coco_cate_val2014.csv" \
    --box-csv="./data/coco_box_val2014.csv" \
    --train=False
```

Re-create a single annotation / label table for YOLO training:

```sh
# insert training labels
pdm run yolov3 create-labels --train=True
# insert validation labels
pdm run yolov3 create-labels --train=False
```

Train model:

```sh
pdm run yolov3 train --seed=0 --n-epoch=20000 --lr=0.001 \
    --batch-train=16 --batch-valid=16 --eval-span=1000
```

Fine-tuning pre-trained model:

```sh
pdm run yolov3 tuning --path-params=data/model_yolov3_params.pickle \
    --path-states=data/model_yolov3_states.pickle \
    --seed=0 --n-epoch=10000 --lr=0.00001 \
    --batch-train=16 --batch-valid=16 --eval-span=16
```
