## part 1. Introduction

Implementation of a simple pipeline for training mnist in Tensorflow (TF-Slim).<br>
- [x] A simple neural network architecture.
- [x] Basic working demo
- [x] Training pipeline

The network architecture shows that I am free to set up, you can modify it according to your needs.

## part 2. Quick start
1. Clone this file
```bashrc
$ git clone https://github.com/po1234263/TFslim-mnist.git
```
2.  You are supposed  to install some dependencies before getting out hands with these codes.
```bashrc
$ cd TFslim-mnist
$ pip install -r ./docs/requirements.txt
```
3. We provide some `.ckpt` files in the dir `./tmp`, and you can use this file to run the demo script
```bashrc
$ python test.py
```
## part 3. Train by yourself
#### how to train it ?
```
$ python train.py
$ tensorboard --logdir=your_log_path
```
As you can see in the tensorboard, if you train for too long, the model starts to overfit and learn patterns from training data that does not generalize to the test data.
