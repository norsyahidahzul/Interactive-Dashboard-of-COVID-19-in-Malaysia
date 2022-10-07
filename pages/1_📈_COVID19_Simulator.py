import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from tabulate import tabulate
import plotly.graph_objects as go
import plotly.io as pio

def COVID19_Simulator(): 
   
## sidebar
# Using "with" notation
 with st.sidebar:
    model = st.radio(
        "Choose a type of SEIRD model",
        ("Classical", "Modified")
    )

 if model == 'Classical':
    st.write('**You have selected Classical SEIRD model.**')
   
 else:
    st.write('**You have selected Modified SEIRD model.**')

st.set_page_config(page_title="COVID-19 Simulator", page_icon="📈")
st.markdown("# COVID-19 Simulator")
st.sidebar.header("COVID-19 Simulator")
st.write(
    """This simulator shows the effects of considered factors upon its increment or decrement in values. You may try to chose 
    the values of parameters and see the changes of the SEIRD simulation. Enjoy!"""
)

COVID19_Simulator()

