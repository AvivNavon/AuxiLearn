# AuxiLearn for generating auxiliary task

To use AuxiLearn for learning to generate tasks on Oxford-IIIT Pet dataset:
1. Create a directory for storing the dataset:
```bash
cd ./experiments/oxford_pet
mkdir dataset
```
2. Download data and meta-data from [here](https://www.robots.ox.ac.uk/~vgg/data/pets/), and store it on ./dataset.
3. Run the following for distributing the data between train\val\test folders
```bash
python data.py
```
4. Run AuxiLearn
```bash
python trainer.py
```