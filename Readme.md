# Keras model retraining 

This project offers funtionality for training a top model over a pre-loaded keras model, fine tuning some layers of a 
pre-loaded model and retraining a pre-loaded model. There is also an option to train a simple CNN from scratch. The
keras models supported for fine-tuning/retraing are:
  * [VGG16](https://keras.io/applications/#vgg16)
  * [VGG19](https://keras.io/applications/#vgg19)
  * [ResNet-50](https://keras.io/applications/#resnet50)
  * [Inception-v3](https://keras.io/applications/#inceptionv3)
  
The TransferModel class in src/keras_transfer_learning.py wraps model functionality and src/train_helper.py offers a
training suite.

It is recommended to keep data specific code in seperate scripts. See branch planet-amazon-kaggle-src for an example of
using the retraining functionality on data of the 
[Planet: Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space) 
kaggle competition.

Make sure to use python 3, some code is not compatible with python 2

This project was inspired from:  
[https://github.com/EKami/planet-amazon-deforestation]()  
I recommend it for a wonderful jupyter notebook breakdown of training a CNN on competition data.

##### Requirements  
    matplotlib==2.0.2
    Keras==2.0.6
    numpy==1.13.1
    tensorflow_gpu==1.2.1
    pandas==0.20.1
    h5py==2.7.0
    kaggle_data==0.1
    tqdm==4.14.0
    Pillow==4.2.1
    seaborn==0.8
    scikit_learn==0.19b2
