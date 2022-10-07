import streamlit as st
import numpy as np


def COVID19_Simulator(): 
   
## sidebar
# Using "with" notation
 with st.sidebar:
    model = st.radio(
        "Choose a type of SEIRD model",
        ("Classical", "Modified")
    )

 if model == 'Classical':
    st.write('You have selected Classical SEIRD model.')
 else:
    st.write('You have selected Modified SEIRD model.')

st.set_page_config(page_title="COVID-19 Simulator", page_icon="📈")
st.markdown("# COVID-19 Simulator")
st.sidebar.header("COVID-19 Simulator")
st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

COVID19_Simulator()

