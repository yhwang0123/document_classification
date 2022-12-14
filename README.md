# Multi-class Text Classification

## Objective: 
The objective of this project is to classify the technical text to multi-class. 

## Dataset: 
* We have a metadata with 8 files regarding all the labels. Meanwhile, we have a dataset containing more than 500 technical text with 50+ labels. One text belongs to one label. The languages of the text are both in French and Dutch, while the majority language is French. 

## Workflow
![alt text](https://github.com/yhwang0123/document_classification/blob/main/assets/workflow.png)

### Data Exploration
We dropped the duplicated data, and selected the data only from the source of training. After that, we found the dataset is extremly unbalanced, with 1 or 2 samples in a certain labels, while 20+ samples in another label.

### Data Preprocessing and Text Argumentation

For handlining the imbalance in the classes we use text augmentation in two steps: 

- Back Translation: \
We generated synthetic data by back-translation. We used the GoogleTrans library in four different languages, English, Spanish, Italian, and Chinese and we made back translation to French. Afterwards, we added the translation for minority labels to increase the samples. Then we dropped the duplicates.
As the number of Dutch text are very limited, we translated them in French with google translation. So the language of the text are consistent.

- Text Augmentation: \
We applied contextual text augmentation using the NLPAUG library, and we dropped duplicates text again. 

## Models:
We applied altogether four models in machine learning and deep learning, to make the text classification.
1. Machine Learning
- Logistic Regression
- SVM (Support Vector Machine)
- Naive Bayes

2. Deep learning
- Bert \
As Bert is mainly for english text classification, we translated all the text (both French and Dutch) into English with the GoogleTrans library, and applied the pretrained model 'distilbert-base-uncased' to train the model.


## Evaluation of models:
* Model accuracy comparison\
![alt text](https://github.com/yhwang0123/document_classification/blob/main/assets/performance.png)

## App Deployment

We developed an app using Streamlit. On this app, users can upload dataset in json or csv format, and choose a model from machine learning or deep learning and get the class prediction for the text. 
![alt text](https://github.com/yhwang0123/document_classification/blob/main/assets/streamlit.PNG)

## Required library
To run the code in this repo, you need to install libraries as below:
- `pip install -r requirements.txt`

## Authors of this project
* [Pragati Khadka](https://github.com/PragatiKhadka)
* [Yuri Hernandez Flores](https://github.com/YuriHFlowers)
* [Yihui Wang](https://github.com/yhwang0123)