# NYUv2 Experiment

<p align="center"> 
    <img src="https://github.com/AvivNavon/AuxiLearn/resources/nyu_losses_and_gradients_tight.png" width="800">
</p>

## Code and Dataset 

We use some of the code implementation (e.g. for SegNet) and the data provided in [MTAN](https://github.com/lorenmt/mtan). 
Please download the dataset provided [here](https://www.dropbox.com/s/p2nn02wijg7peiy/nyuv2.zip?dl=0). It consists of the following tasks: semantic segmentation, depth, surface normal. 
We use the 13-classes semantic segmentation as the main task.

## Baselines 

To run baselines:

```bash
python3 trainer_baseline.py --dataroot <dataroot> --method <method>
```

Where `method` is one of:

|method|desc|
|----|----|
|stl|Single task learning |
|dwa|[Dynamic weighted average](https://arxiv.org/pdf/1803.10704.pdf)|
|uncert| [Weight uncertainty](https://arxiv.org/abs/1705.07115)|
|cosine| [Gradient Cosine Similarity](https://arxiv.org/abs/1812.02224)|
|gradnorm| [GradNorm](https://arxiv.org/abs/1711.02257)|
|equal| Equal weights|

All methods are implemented in `experiments/weight_methods.py`.

## Ours

### Auxiliary Networks

We 3 variants of AuxiLearn:

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
