# Digit-Recognizer

#### Python Libraries Needed

* numpy scipy matplotlib
* pandas==0.18.1
* scikit-learn==0.17.1

#### Additional Steps (Requirements) for Neural Network (Step 1 and 2 is optional for training on GPU)

1. Install [cuda](https://developer.nvidia.com/cuda-toolkit)(v7.5)
2. Install [theano](http://deeplearning.net/software/theano/)==0.8.2 [Windows Install](http://deeplearning.net/software/theano/install_windows.html)or [Ubuntu Install](http://deeplearning.net/software/theano/install_ubuntu.html), [GPU Config](http://deeplearning.net/software/theano/tutorial/using_gpu.html)
3. Add python library h5py==2.6.0
4. Add python library keras==1.0.2

Traun Leyden kindly provides instructions on how to install cuda on amazon ec2  [here](http://tleyden.github.io/blog/2015/11/22/cuda-7-dot-5-on-aws-gpu-instance-running-ubuntu-14-dot-04/)

#### Files and Descriptions

1. cnn1.py = Simple Convolutional Nerual Network
2. cnn2.py = Multi Convolutional Layer Nerual Network

#### Performance Measure

| Model         | Offline Cost (secs) | Online Cost (secs) | Accuracy(%) |
| ------------- |--------------------:| ------------------:|------------:|
| PCA + LDA                                 | 20.8 | 3.5 | 84.2 |
| PCA + Linear Support Vector Machine       | 1450.5 | 90.9 | 90.1 |
| Random Forest + AdaBoost                  | 368.1 | 26.2 | 96.6 |
| Simple Convolutional Nerual Network*      | 42 | 7 | 98.8 |
| Multi Convolutional Layer Nerual Network* | 4320 | 18 | 99.5 |

This model runs on GPU

Last Update on 05/12/2016
