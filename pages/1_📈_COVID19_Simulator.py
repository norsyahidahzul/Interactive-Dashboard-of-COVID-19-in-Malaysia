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
   description ='''
   See how the value of epidemiological parameters affects COVID-19 cases. 
   Move the slider to change the parameters to display
   '''
   sliderCaption='Select Infection rate (β) for the chart'
   beta=st.sidebar.slider("Infection rate (β)", min_value=0.0, max_value=100.0, value=3.0, step=0.1)
   sigma=st.sidebar.slider("Incubation rate (σ)", min_value=0.0, max_value=100.0, value=0.2, step=0.1)
   gamma=st.sidebar.slider("Recovery rate (γ)", min_value=0.0, max_value=100.0, value=0.2, step=0.1)
   mu=st.sidebar.slider("Death rate (μ)", min_value=0.0, max_value=100.0, value=0.2, step=0.1)

   st.write('**You have selected Classical SEIRD model simulator :smile:.**')
   st.subheader("**Classical SEIRD model simulator**")
  
   if st.button('Click here for more info on its mathematical expressions'):
         st.write('The mathematical expression of classical SEIRD model in terms of ordinary differential equations (ODE) can be expressed as follows:')
         st.latex(r'''\frac{ds}{dt} = -\frac{\beta S\left(t\right)I\left(t\right)}{N}''')
         st.latex(r'''\frac{dE}{dt} = \frac{\beta S\left(t\right)I\left(t\right)}{N} ''')
         st.latex(r'''\frac{dI}{dt} = \sigma E\left(t\right) - \gamma I\left(t\right)- \mu I\left(t\right)''')
         st.latex(r'''\frac{dR}{dt} =  \gamma I\left(t\right)''')
         st.latex(r'''\frac{dD}{dt} =  \delta R\left(t\right)''')
             
   
 else:
   
   beta_I=st.sidebar.slider("Infection rate (βI)", min_value=0.0, max_value=100.0, value=3.0, step=0.1)
   beta_E=st.sidebar.slider("Infection rate (βE)", min_value=0.0, max_value=100.0, value=3.0, step=0.1)
   sigma=st.sidebar.slider("Incubation rate (σ)", min_value=0.0, max_value=100.0, value=0.2, step=0.1)
   gamma=st.sidebar.slider("Recovery rate (γ)", min_value=0.0, max_value=100.0, value=0.2, step=0.1)
   mu=st.sidebar.slider("Death rate (μ)", min_value=0.0, max_value=100.0, value=0.2, step=0.1)
   delta=st.sidebar.slider("Reinfection rate (δ)", min_value=0.0, max_value=100.0, value=0.2, step=0.1)
   
   st.write('**You have selected Modified SEIRD model simulator :smile:.**')
   st.subheader("**Modified SEIRD model simulator**")
   
   if st.button('Click here for more info on its mathematical expressions'):
         st.write('The mathematical expression of modified SEIRD model in terms of ordinary differential equations (ODE) can be expressed as follows:')
         st.latex(r'''\frac{ds}{dt} = -\frac{\beta_I\left(t\right)S\left(t\right)I\left(t\right)}{N} -\frac{\beta_E\left(t\right)S\left(t\right)E\left(t\right)}{N} 
         + \delta\left(t\right)R\left(t\right)''')
         st.latex(r'''\frac{dE}{dt} = \frac{\beta_I\left(t\right)S\left(t\right)I\left(t\right)}{N} +\frac{\beta_E\left(t\right)S\left(t\right)E\left(t\right)}{N} 
          - \sigma\left(t\right)E\left(t\right)''')
         st.latex(r'''\frac{dI}{dt} = \sigma\left(t\right)E\left(t\right) - \gamma\left(t\right)I\left(t\right)- \mu\left(t\right)I\left(t\right)''')
         st.latex(r'''\frac{dR}{dt} =  \gamma\left(t\right)I\left(t\right)- \delta\left(t\right)R\left(t\right)''')
         st.latex(r'''\frac{dD}{dt} =  \delta\left(t\right)R\left(t\right)''')
   

st.set_page_config(page_title="COVID-19 Simulator", page_icon="📈")
st.markdown("# COVID-19 Simulator")
st.sidebar.header("COVID-19 Simulator")
st.write(
    """This simulator shows the effects of considered factors upon its increment or decrement in values. You may try to chose 
    the values of parameters and see the changes of the SEIRD simulation. Enjoy!"""
)

COVID19_Simulator()

