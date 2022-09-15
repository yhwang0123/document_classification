# Multi-class Text Classification

## Objective: 
The objective of this project is to classify the technical text to multi-class. 

## Dataset: 
* We have a metadata with 8 files regarding all the labels. Meanwhile, we have a dataset containing more than 500 technical text with 50+ labels. One text belongs to one label. The languages of the text are both in French and Dutch, while the majority language is French. 

## Workflow
![alt text](https://github.com/yhwang0123/document_classification/blob/main/assets/workflow.png)

### Data Explore
We dropped the duplicated data, and selected the data only from the source of training. After that, we found the dataset is extremly unbalanced, with 1 or 2 samples in a certain labels, while 20+ samples in another label.

### Data Preprocessing and Text Argumentation
- Translation:
In order to add samples to minority label, and make data more balanced, we used goolge translate api to translate the original sample to other languages(English, Spanish, Portuguese,Chinese),and translate back to French.

As the number of Dutch text is very limited, we use google translate to make it in French.

- Text Argumentation


## Models:
We have applied several models in machine learning and deep learning, to make the text classification.
1. Machine Learning
- Logistic Regression
- SVM
- Naive Bay

2. Deep learning
- Bert
As Bert is mainly for english text classification, we translated all the text (both French and Dutch) into English, and applied the pretrained model 'distilbert-base-uncased' to train the model.


## Evaluation of models:
1. Machine learning model 
- SVM Accuracy Score ->  91.52542372881356
- Naive Bay Accuracy Score ->  0.8559322033898306
- Logistic regression Accuracy Score ->  0.8728813559322034

2. Deep learning model
Val Accuracy: 0.88

## App Deployment

We developed an app using Streamlit. On this app, user can upload dataset in json format or csv format, and choose the model (machine learning or deep learning) and get the class prediction for the text. 


## Authors of this project
* [Pragati Khadka](https://github.com/PragatiKhadka)
* [Yuri Hernandez Flores](https://github.com/YuriHFlowers)
* [Yihui Wang](https://github.com/yhwang0123)
 