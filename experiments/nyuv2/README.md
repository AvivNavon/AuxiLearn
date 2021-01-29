# NYUv2 Experiment

<p align="center"> 
    <img src="https://github.com/AvivNavon/AuxiLearn/blob/master/resources/nyu_losses_and_gradients_tight.png" width="800">
</p>

## Code and Dataset 

We use some of the code implementation (e.g. for SegNet) and the data provided in [MTAN](https://github.com/lorenmt/mtan). 
Please download the dataset provided by [_End-to-End Multi-Task Learning with Attention_](https://arxiv.org/pdf/1803.10704.pdf). 
The data is available [here](https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0) (2.92 GB). 
It consists of the following tasks: semantic segmentation, depth, surface normal. 
We use the 13-classes semantic segmentation as the main task.

### Download dataset

Download the data from [here](https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0), or run:

```bash
wget https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0
```

Unzip the data:

```bash
unzip nyuv2.zip?dl=0
```

This will create a folder `nyuv2`.

### Validation split

We preallocated 10\% of training examples to construct a validation set. 
The mapping of indices for train/val is available in `experiments/nyuv2/hpo_validation_indices.json`.

## Baselines 

To run baselines:

```bash
python3 trainer_baseline.py --dataroot <dataroot> --method <method>
```

where `method` is one of:

|method|desc|
|----|----|
|stl|Single task learning |
|equal| Equal weights|
|dwa|[Dynamic weighted average](https://arxiv.org/pdf/1803.10704.pdf)|
|uncert| [Weight uncertainty](https://arxiv.org/abs/1705.07115)|
|cosine| [Gradient Cosine Similarity](https://arxiv.org/abs/1812.02224)|
|gradnorm| [GradNorm](https://arxiv.org/abs/1711.02257)|

All methods are implemented in `experiments/weight_methods.py`.

## Ours

### Auxiliary Networks

Three variants of AuxiLearn:

|auxiliary net|desc|
|----|----|
|Linear|linear weights over loss terms |
|Nonlinear| NN over loss terms|
|ConvNet| ConvNet applied to the per-pixel losses|

To run the linear/nonlinear auxiliary network use: 

```bash
python3 trainer.py --dataroot <dataroot> --aux-net <linear/nonlinear>
```

And for the ConvNet please run:

```bash
python3 trainer_cnn.py --dataroot <dataroot>
```
