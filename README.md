# Multi-class Text Classification

## Objective: 
The objective of this project is to classify the technical text to multi-class. One text belongs to one class.

## Dataset: 
* We have a dataset containing more than 500 technical text with 50+ labels. The languages of the text are both in French and Dutch, while the majority language is French. 

## Workflow
![alt text](https://github.com/yhwang0123/document_classification/blob/main/workflow.png)

### Data Explore
1. we dropped the duplicated data, and select the data only from the source of training. After that, we found the dataset is extremly unbalanced, with 1 sample in a certain label, while 20+ samples in another label.

### Data Preprocessing and Text Argumentation
In order to add samples to minority label, we use goolge translate api, to translate the original sample to other languages(English, Spanish, Portuguese,Chinese),and translate back to French. 

For the number of Dutch text is very limited, we use google translate to make it in French.


## Models:
We have created applied severl models in machine learning and deep learning:
1. Machine Learning
- Logistic Regression
- SVM
- Naive Bay

2. Deep learning
- Bert
as Bert is only for english text classification, we translate the French text in the dataset to English, and apply the pretrained model 'distilbert-base-uncased' to train the model


## Evaluation of models:
1. Machine learning model 
- SVM Accuracy Score ->  91.52542372881356
- Naive Bay Accuracy Score ->  0.8559322033898306
- Logistic regression Accuracy Score ->  0.8728813559322034

2. Deep learning model
Accuracy for test: 0.88

## App Deployment

we develop an app using Streamlit. On this app, user can upload dataset in json format or csv format, and choose the model (machine learning or deep learning) and get the class prediction for the text. 


## Authors of this project : 
* [Pragati Khadka](https://github.com/PragatiKhadka)
* [Yuri Hernandez Flores](https://github.com/YuriHFlowers)
* [Yihui Wang](https://github.com/yhwang0123)
 