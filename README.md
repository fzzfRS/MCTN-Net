## MCTN-Net:AMulti-Class Transportation Network Extraction Method Combining Orientation and Semantic Features ##

## Requirements
* [PyTorch](https://pytorch.org/) (version = 1.10.2)
* [sknw](https://github.com/yxdragon/sknw)
* [networkx](https://networkx.github.io/) (version = 2.4)
* json
* skimage
* numpy
* tqdm

## Data Preparation


*Download MCTNDataset:

BaiduNetdisk: [MCTNDataset](https://pan.baidu.com/s/1ET1L5aZgg8El-dHmhhuGdQ) (Access code: x1qp )

* Organize the data as the following tree structure.

```
data/MCTNDataset
|   train.txt
|   val.txt
|   test.txt

|
└───train
│   └───gt
│   └───images
└───val
│   └───gt
│   └───images
└───test
│   └───gt
│   └───images

```

## Training

Train Multi-Task learning framework to predict segmentation and orientation.

__Training MTL Help__
```
usage: train_mtl.py [-h] --config CONFIG
                    --model_name {LinkNet34MTL,StackHourglassNetMTL}
                    --dataset {deepglobe,spacenet}
                    --exp EXP
                    [--resume RESUME]
                    [--model_kwargs MODEL_KWARGS]
                    [--multi_scale_pred MULTI_SCALE_PRED]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       config file path
  --model_name 			{MCTNNet}
                        Name of Model = ['MCTNDataset']
  --exp EXP             Experiment Name/Directory
  --resume RESUME       path to latest checkpoint (default: None)
  --dataset 			{MCTNDataset}
                        (default: MCTNDataset)
  --model_kwargs 		MODEL_KWARGS
                        parameters for the model
  --multi_scale_pred 	MULTI_SCALE_PRED
                        perform multi-scale prediction (default: True)
```

__Sample Usage__

* Training with MCTN-Net
```
CUDA_VISIBLE_DEVICES=0 python train.py --config config.json --dataset MCTNDataset --model_name "MCTNNet" 
```

## Test

* Test with MCTN-Net
```
CUDA_VISIBLE_DEVICES=0 python test.py --config config.json --dataset MCTNDataset --model_name "MCTNNet" 
```

## Acknowledgments
Code is implemented based on [Road-connectivity](https://github.com/anilbatra2185/road_connectivity) and [CoordAttention](https://github.com/houqb/CoordAttention).
