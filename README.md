# digit-recognizer

# Files and description
1. cnn1.py = Simple Convolutional Nerual Network
2. cnn2.py = Multi Convolutional Layer Nerual Network

# Performance Measure
| Model         | Offline Cost (secs) | Online Cost (secs) | Accuracy(%) |
| ------------- |--------------------:| ------------------:|------------:|
| PCA + LDA                                 | 20.81 | 3.5 | 84.2 |
| PCA + Linear Support Vector Machine       | 145.49 | 90.9 | 90.1 |
| Random Forest + AdaBoost                  | 368.1 | 26.2 | 96.6 |
| Simple Convolutional Nerual Network*      | 42 | 7 | 98.8 |
| Multi Convolutional Layer Nerual Network* | 4320 | 18 | 99.5 |
* This model runs on GPU
