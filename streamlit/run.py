import streamlit as st
import pandas as pd
import numpy as np
import time

from functions import get_csv_download_link , get_txt_download_link, augment_sentence
from functions import *






st.title('NLP Data Augmentation')



st.sidebar.title('Augmentation Configuration')
interaction_type_input = st.sidebar.selectbox(
        'select the interaction type',
    ('static', 'interactive')
)

# if static mode is selected
if interaction_type_input == 'static':
    st.sidebar.write('Static mode is selected. Please upload the file and click on the button to start augmentation.')
    # select the file type
    file_type_input = st.sidebar.selectbox('select the file type', ('csv', 'txt'))
    if file_type_input == 'csv':
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df = df[['class','text']]

            st.write('Dataframe:')
            st.write(df.head())
            augment_method = st.selectbox('select the augmentation method',
                        ('eda_augmenter','aeda_augmenter', 'wordnet_augmenter', 'clare_augmenter',
                        'backtranslation_augmenter'))            
            pct_words_to_swap = st.slider('percentage words to swap: ', 0.0, 1.0, 0.5)
            transformations_per_example = st.slider('transformations per example: ', 0, 10, 1)
            fraction = st.slider('fraction of data to augment: ', 0.0, 1.0, 0.5)
            keep_original = st.checkbox('keep original')
            
            if st.button('Start Augmentation'):
                with st.empty():
                    st.write('Augmentation started. Please wait for the results.')
                    res = augment_text(df, augment_method, pct_words_to_swap, transformations_per_example, fraction, keep_original)
                    st.empty()
                st.write('Augmentation completed. Please download the file.')
                st.markdown(get_csv_download_link(res), unsafe_allow_html=True)

    elif file_type_input == 'txt':
        uploaded_file = st.sidebar.file_uploader("Choose a TXT file", type="txt")
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            st.write('Dataframe:')
            st.write(df.head())
            augment_method = st.selectbox('select the augmentation method',
                        ('eda_augmenter','aeda_augmenter', 'wordnet_augmenter', 'clare_augmenter',
                        'backtranslation_augmenter'))            
            pct_words_to_swap = st.slider('percentage words to swap: ', 0.0, 1.0, 0.5)
            transformations_per_example = st.slider('transformations per example: ', 0, 10, 1)
            fraction = st.slider('fraction of data to augment: ', 0.0, 1.0, 0.5)
            keep_original = st.checkbox('keep original')
            
            if st.button('Start Augmentation'):
                with st.empty():
                    st.write('Augmentation started. Please wait for the results.')
                    res = augment_text(df, augment_method, fraction, pct_words_to_swap,transformations_per_example, include_original=keep_original)
                    st.empty()
                st.write('Augmentation completed. Please download the file.')
                st.markdown(get_txt_download_link(res), unsafe_allow_html=True)
            
           
    else:
        pass
    
        

# if interactive mode is selected
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
            if st.button('Start Augment'):
                with st.empty():
                    st.write('Augmentation started. Please wait for the results.')

                    res = augment_sentence(sentence_input,augment_method,pct_words_to_swap,transformations_per_example)
                    
                    st.write('Augmentation completed. \n augemented sentences are: \n')
                for i in res:
                    time.sleep(0.5)
                    st.write(i)
    
    # todo: work on options for interactive mode for file
    #else:



                    
