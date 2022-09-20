import streamlit as st
import pandas as pd
import numpy as np
import time



st.title('Data Augmentation Comparison')



st.sidebar.title('configurationa')
interaction_mode_input = st.sidebar.selectbox(
    'select the interaction mode',
    ('static', 'interactive')
)
if interaction_mode_input == 'static':
    interaction_mode = False
else:
    interaction_mode = True



