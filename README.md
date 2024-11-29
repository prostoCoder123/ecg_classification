# ecg_classification

A couple of neural networks to classify people by their ecg

I'm a beginner in ML and Python and set the goal to put theory into practice and therefore published this code.

# -----

What if it is possible to recognize of people from their ecgs? Is it real that an ecg can be used to identify a person because it contains special traits of the person's heart activity?

In the language of the AI-specialists, this task can be formulated as **multiclass single-label classification task**, where each input sample should be categorized into more than two categories and each input sample can be assigned only one label. In this case the classes are associated with people and the labels are the their ids.

To implement this classification task with Machine Learning methods we can build a model and train it (supervised learning) to get desired outputs from given inputs. In deep learning, neural networks, structured in layers stacked on top of each other, do this job. An ANN-model gets inputs and transforms them to outputs by calculating some kind of function. The problem is to find the most appropriate function resulted with the maximum accuracy.

To build a ML-model we need to preprocess raw data before feeding it into a neural network. We should provide a bunch of samples for each class so that the model could learn to distiguish them (the more samples for each class, the better).

First step is to highlight the most characteristic cyclic parts of an ecg record that include special traits of the person's heart activity - these are **QRS-complexes**.
Second step is to find an ecg dataset and preprocess the data.

# Data preprocessing

The PTB XL dataset was chosen for experimenting. The data processing includes the following steps: 
1) download the data (data_processing/data_loader.ipynb)
2) extract the raw ecg data with the labels
3) parse the ecg waveforms into fixed-length segments with qrs-complexes
4) prepare the data to be fed to the proposed ML-models and also to be used for evaluation of the results

# Model training and evaluation

A Neural network model is a stack of layers. To build such kind of thing, one needs to decide with the architecture of those layers and after that considers the parameters for the layers and training process.
Convolutional neural networks (CNN) are known as the really effective methods for image classification. Recurrect neural networks are good at text processing and time series forecasting.
In this case the CNN model based on Resnet architecture was proposed to perform ecg classification task.

# How to use

1) Run the script to download the data:    /data_processing/ptb_xl_data_loader.ipynb
2) Run the script to preprocess the data:  /data_processing/ptb_xl_data_processing.ipynb
3) Run the script to train the model:      train_ml_models.ipynb


# Results

| Accuracy  | Loss   | Val-accuracy | Val-loss | Test accuracy | Test loss | Epochs |
| :-------- | :----- | :----------- | :------- | :------------ | :-------- | :----- |
| 0.9646    | 0.1093 | 0.7520       | 1.8161   | 0.752         | 1.588     | 100    |


