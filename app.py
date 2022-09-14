import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')

siteHeader = st.container()
dataExploration = st.container()
uploadFile = st.container()
modelPrediction = st.container()

@st.cache
def get_data(filename):
  data = pd.read_csv(filename, sep=';')
  return data

@st.cache
def create_dict():
    df_original = get_data('clean_before_translation1409.csv')
    X = df_original["text"].tolist()
    y = pd.get_dummies(df_original['label'])
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

def make_prediction(text, dict):
    # load the model from disk
    text = text.lower()
    ml_model = joblib.load('logistic_regression_new.sav')
    prediction = ml_model.predict([text])
    return dict[prediction[0]]

def make_prediction2(text, dict):
    # load the model from disk
    ml_model = joblib.load('logistic_regression_model.sav')
    prediction = ml_model.predict([text])
    return dict[prediction[0]]    

with siteHeader:
  st.title('Welcome to NLP Project!')

with dataExploration:
    st.header('Data Exploration')
    data = get_data('clean_before_translation1409.csv')
    label_dist = pd.DataFrame(data['label'].value_counts())   
    st.bar_chart(label_dist)

with uploadFile:
    uploaded_file = st.file_uploader("Choose a file")    
    if uploaded_file is not None:
        dataframe = pd.read_json(uploaded_file)
        dataframe = pd.DataFrame(dataframe["text"])
        st.write(dataframe)
        texts = dataframe["text"].tolist()

with modelPrediction:
    st.header('Let\'s make some prediction')
    ml = st.button('ML Model') 
    if ml:
        #st.write(texts[0])
        for text in texts:
            # text = "Eclairage de la gaine , à compléter rappel : 2"
            cleaned_text = clean_text(text)
            dict_labels = create_dict()
    
            prediction = make_prediction(cleaned_text, dict_labels)
            #st.write("The text belongs to class: ",prediction)
            dataframe['Prediction1'] = prediction
    
            prediction2 = make_prediction2(cleaned_text, dict_labels)
            #st.write("No lowercase: The text belongs to class: ",prediction2)
            dataframe['Prediction2'] = prediction2

        st.write(dataframe)    


        # testing with csv file
        # with open('test_data.csv', encoding="utf8", errors='ignore') as f:
        #     data = pd.read_csv(f, sep=';')
        #     texts = data['text'].tolist()
        #     st.write(texts[0])
        #     for text in texts:

        #         # text = "Eclairage de la gaine , à compléter rappel : 2"
        #         st.write("Text for classification: ",text)
        #         cleaned_text = clean_text(text)
        #         dict_labels = create_dict()
        
        #         prediction = make_prediction(cleaned_text, dict_labels )
        #         st.write("The text belongs to class: ",prediction)
        
        #         prediction2 = make_prediction2(cleaned_text, dict_labels )
        #         st.write("No lowercase: The text belongs to class: ",prediction2)
