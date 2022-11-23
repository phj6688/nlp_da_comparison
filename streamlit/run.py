import streamlit as st
import pandas as pd
import numpy as np
import time

from functions import get_table_download_link , augment_sentence
from functions import *






st.title('NLP Data Augmentation')



st.sidebar.title('Augmentation Configuration')
interaction_mode_input = st.sidebar.selectbox(
        'select the interaction mode',
    ('static', 'interactive')
)


if interaction_mode_input == 'static':
    st.sidebar.write('Static mode is selected. Please upload the file and click on the button to start augmentation.')
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)
        st.sidebar.write('Augmentation started. Please wait for the results.')
        time.sleep(5)
        st.sidebar.write('Augmentation completed. Please download the file.')
        st.sidebar.markdown(get_table_download_link(df), unsafe_allow_html=True)

else:
    interaction_interactive_input = st.sidebar.selectbox(
    'select the mode',
    ('setence', 'file')
    )
    if interaction_interactive_input == 'setence':
        st.write('Interactive mode is selected. Please enter the sentence and click on the button to start augmentation.')
        sentence_input = st.text_input('Enter the sentence')
        if sentence_input is not None:
            augment_method = st.selectbox('select the augmentation method',
                        ('eda_augmenter','aeda_augmenter', 'wordnet_augmenter', 'clare_augmenter',
                        'backtranslation_augmenter'))            
            pct_words_to_swap = st.slider('percentage words to swap: ', 0.0, 1.0, 0.5)
            transformations_per_example = st.slider('transformations per example: ', 0, 10, 1)
            if st.button('Augment'):
                with st.empty():
                    st.write('Augmentation started. Please wait for the results.')

                    res = augment_sentence(sentence_input,augment_method,pct_words_to_swap,transformations_per_example)
                    time.sleep(2)
                    st.write('Augmentation completed. \n augemented sentences are: \n')
                for i in res:
                    time.sleep(0.5)
                    st.write(i)
                    


                    
