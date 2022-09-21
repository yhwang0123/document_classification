import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
#from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
import googletrans
from googletrans import Translator
import numpy as np
import time 

# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('omw-1.4')

siteHeader = st.container()
uploadFile = st.container()
modelPrediction = st.container()
ml_prediction = st.container()
dl_prediction = st.container()

def get_data(filename, sep=';'):
  data = pd.read_csv(filename, sep=sep)
  return data

def create_dict():
    df_original = get_data('./model/Bert_training.csv', sep=',')
    X = df_original["text"].tolist()
    y = pd.get_dummies(df_original['code'])
    dict = {i: name for i, name in enumerate(y.columns)}

    return dict

special_character_remover = re.compile('[/(){}\[\]\|@,;:]')
extra_symbol_remover = re.compile('[^0-9a-z #+_]')
STOPWORDS = nltk.corpus.stopwords.words('french')

def clean_text(text):
    #text = text.lower()
    text = special_character_remover.sub(' ', text)
    text = extra_symbol_remover.sub('', text)
    text = ''.join(c for c in text if not c.isdigit())
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  

    # remove any spaces
    text = text.strip()
    # remove any white spaces from beginning of string
    text = text.lstrip() 
    # remove any white spaces from ending of string
    text = text.rstrip()
    text = re.sub('\s+', ' ', text)
    #removing : \ characters  from the text
    text = re.sub(r'(:\S+) | (\S+)', r'', text)
    return text   

def make_prediction(text, dictionary, ml_model):
    #text = text.lower()
    prediction = ml_model.predict([text])
    return dictionary[prediction[0]]

with siteHeader:
  st.title('Welcome to NLP Project!')

with uploadFile:
    uploaded_file = st.file_uploader("Please upload a file in json/csv format")  
    if uploaded_file is not None and uploaded_file.type=='application/json':
        dataframe = pd.read_json(uploaded_file)
        texts = dataframe["text"].tolist()

    if uploaded_file is not None and uploaded_file.type=='text/csv':   
        dataframe = pd.read_csv(uploaded_file, sep=',', encoding='latin-1')
        texts = dataframe["text"].tolist() 

with modelPrediction:
    ml = st.button('ML Model') 
    dl = st.button('Deep Learning Model') 
    dict_labels = create_dict()

with ml_prediction:   
    if ml:
        time.sleep(2)
        st.subheader('Result from Machine Learning algorithm')
        dataframe['svm_label'] = ''    
        for text in texts:
            cleaned_text = clean_text(text)    
            svm_model = joblib.load('./model/SVM.sav')
            svm_pred_label = make_prediction(cleaned_text, dict_labels, svm_model)
            dataframe.loc[dataframe['text'] == text, "svm_label"] = svm_pred_label

        st.write(dataframe)  

with dl_prediction: 
    if dl:        
        st.subheader('Result from Deep Learning algorithm')
        dataframe['bert_label'] = '' 
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')   
        new_model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=50)
        new_model.load_weights('./model/final_model.h5')   
        # use translate method to translate a string 
        translator = Translator()
        for text in texts:
            org_text = text
            translated = translator.translate(text, dest='en')
            text = translated.text
            text = clean_text(text)
            encodings = tokenizer([text], max_length=200, truncation=True, padding=True)
            ds = tf.data.Dataset.from_tensor_slices(dict(encodings))
            predictions = new_model.predict(ds)
            bert_pred_label = dict_labels[np.argmax(predictions[0])]
            st.write("The text with class: ", org_text,":", bert_pred_label)
            dataframe.loc[dataframe['text'] == text, "bert_label"] = bert_pred_label
        
        st.write(dataframe)