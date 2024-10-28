# RBE 577 - Homework 2<br /><br />Convolutional Neural Networks for Autonomous Vehicles
### 
### Azzam Shaikh
### Oct. 14, 2024

The purpose of this assignment is to finetune a pretrained ResNet model to classify images of road vehicles. This network would provide autonomous vehicles the ability to classify surrounding vehicles using on-board vision systems. 

The dataset used for this project can be found via [Kaggle](https://www.kaggle.com/datasets/marquis03/vehicle-classification/data). The dataset contains 1800 images of various vehicles and is split into a train dataset with 1400 images, a validation dataset with 200 images, and a test dataset with 200 images. 

## Dependencies

The dependencies for this project are included in the `requirements.txt` file.

The project utilizes Python 3.12.4 with the following packages and verisons:

- numpy==1.26.4
- matplotlib==3.8.4
- torch==2.4.0+cu124
- torchvision==0.19.0+cu124
- torchsummary==1.5.1
- tensorboard==2.17.1
- tensorboard-data-server==0.7.2

To install the required dependencies, run the following command:

```
pip install -r requirements.txt
```

## Running the Model

The `src` folder contains the script that contains all the code developed for this project. 

First download the dataset and move the `train`, `val`, and `test` folders to the `dataset` folder located in the `src` folder.

Once complete, run the script. It can be run via an IDE or through the following command:

```
python RBE_577_Homework_2_Script.py
```

## Documentation
The `docs` folder contains a report of the project where the methodologies and results are discussed. 