# AuxiLearn - Auxiliary Learning by Implicit Differentiation

This repository contains the source code to support the paper [_Auxiliary Learning by Implicit Differentiation_](https://arxiv.org/abs/2007.02693), by Aviv Navon*, Idan Achituve*, Haggai Maron, Gal Chechik<sup>†</sup> and Ethan Fetaya<sup>†</sup>, ICLR 2021.

---

<p align="center"> 
    <img src="https://github.com/AvivNavon/AuxiLearn/blob/master/resources/framework.png" width="800">
</p>

## Links

1. [Paper](https://arxiv.org/abs/2007.02693)
2. [Project page](https://avivnavon.github.io/AuxiLearn/)

## Installation

<b>Please note:</b> We encountered some issues and drops in performance while working with different PyTorch versions. Please install AuxiLearn on a clean virtual environment!

```bash
python3 -m venv <venv>
source <venv>/bin/activate
```

On a clean virtual environment clone the repo and install:

```bash
git clone https://github.com/AvivNavon/AuxiLearn.git
cd AuxiLearn
pip install .
```

## Usage

Given a bi-level optimization problem in which the upper-level parameters (i.e., auxiliary parameters) are only 
implicitly affecting the upper-level objective, you can use `auxilearn` to compute the upper-level gradients through implicit differentiation.

The main code component you will need to use is `auxilearn.optim.MetaOptimizer`. It is a wrapper over
PyTorch optimizers that updates its parameters through implicit differentiation.

### Code example

We assume two models, `primary_model` and `auxiliary_model`, and two dataloaders. 
The `primary_model` is optimized using the train data in the `train_loader`, and the `auxiliary_model` is optimized using the auxiliary set in the `aux_loader`.
We assume a `loss_fuction` that return the train loss if `train=True`, or auxiliary set loss if `train=False`.
Also, we assume the training loss is a function of both the primary parameters and the auxiliary parameters, 
and that the loss on the auxiliary set (or validation set) is a function of the primary parameters only. 
In _Auxiliary Learning_, the auxiliary set loss is the loss on the main task (see paper for more details). 

```python
from auxilearn.optim import MetaOptimizer

primary_model = MyModel()
auxiliary_model = MyAuxiliaryModel()
# optimizers
primary_optimizer = torch.optim.Adam(primary_model.parameters())

aux_lr = 1e-4
aux_base_optimizer = torch.optim.Adam(auxiliary_model.parameters(), lr=aux_lr)
aux_optimizer = MetaOptimizer(aux_base_optimizer, hpo_lr=aux_lr)

# training loop
step = 0
for epoch in range(epochs):
    for batch in train_loder:
        step += 1
        # calculate batch loss using 'primary_model' and 'auxiliary_model'
        primary_optimizer.zero_grad()
        loss = loss_func(train=True)
        # update primary parameters
        loss.backward()
        primary_optimizer.step()
        
        # condition for updating auxiliary parameters
        if step % aux_params_update_every == 0:
            # calc current train loss
            train_set_loss = loss_func(train=True)
            # calc current auxiliary set loss - this is the loss over the main task
            auxiliary_set_loss = loss_func(train=False) 
            
            # update auxiliary parameters - no need to call loss.backwards() or aux_optimizer.zero_grad()
            aux_optimizer.step(
                val_loss=auxiliary_set_loss,
                train_loss=train_set_loss,
                aux_params=auxiliary_model.parameters(),
                parameters=primary_model.parameters(),
            )
```

## Citation

If you find `auxilearn` to be useful in your own research, please consider citing the following paper:

```bib
@inproceedings{
navon2021auxiliary,
title={Auxiliary Learning by Implicit Differentiation},
author={Aviv Navon and Idan Achituve and Haggai Maron and Gal Chechik and Ethan Fetaya},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=n7wIfYPdVet}
}
```
