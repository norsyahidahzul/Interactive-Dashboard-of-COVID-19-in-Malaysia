import streamlit as st
import numpy as np

def COVID19_Simulator(): 
   
## sidebar
# Using "with" notation
 with st.sidebar:
    model = st.radio(
        "Choose a type of SEIRD model simulator",
        ("Classical", "Modified")
    )

 if model == 'Classical':
    st.write('**You have selected Classical SEIRD model simulator.**')
    st.latex(r'''\frac{ds}{dt} = -\frac{\beta_I\left(t\right)S\left(t\right)I\left(t\right)}{N} -\frac{\beta_E\left(t\right)S\left(t\right)E\left(t\right)}{N} 
    + \delta\left(t\right)R\left(t\right)''')
   
 else:
    st.write('**You have selected Modified SEIRD model simulator.**')

st.set_page_config(page_title="COVID-19 Simulator", page_icon="📈")
st.markdown("# COVID-19 Simulator")
st.sidebar.header("COVID-19 Simulator")
st.write(
    """This simulator shows the effects of considered factors upon its increment or decrement in values. You may try to chose 
    the values of parameters and see the changes of the SEIRD simulation. Enjoy!"""
)

COVID19_Simulator()

