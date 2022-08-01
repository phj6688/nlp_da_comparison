import streamlit as st
import pandas as pd
import numpy as np



st.title('Data Augmentation Comparison')
st.sidebar.title('sidebar title name !!!!')
interaction_mode_input = st.sidebar.selectbox(
    "select the interaction mode",
    ('static', 'interactive')
)

