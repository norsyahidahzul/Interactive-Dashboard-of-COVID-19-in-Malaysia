import streamlit as st
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from tabulate import tabulate
import plotly.graph_objects as go
import plotly.io as pio
from lmfit import minimize, Parameters, report_fit
pio.renderers.default = "notebook"
%matplotlib inline
plt.style.use('ggplot')

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
    #This is the classical SEIRD model
    #no sporadic cases have been considered (only use one infection rate beta)
    #no reinfection cases (no reinfection rate delta)
    #constant parameters used (beta, sigma, gamma, mu)
    #define Model
   def ode_model(z, t, beta, sigma, gamma, mu):
    S, E, I, R, D = z
    N = S + E + I + R + D
    dSdt = -beta*S*I/N
    dEdt = beta*S*I/N - sigma*E
    dIdt = sigma*E - gamma*I - mu*I
    dRdt = gamma*I
    dDdt = mu*I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]

   #define ODE Solver
   def ode_solver(t, initial_conditions, params):
    initS, initE, initI, initR, initD = initial_conditions
    beta, sigma, gamma, mu= params['beta'].value, params['sigma'].value, params['gamma'].value, params['mu'].value
    res = odeint(ode_model, [initS, initE, initI, initR, initD], t, args=(beta, sigma, gamma, mu)) 
    return res                                   #args used to pass a variable number of arguments to a function



   #initial condition and initial values of parameters
   #initN (Malaysian Population 2020- include non citizen)
   initN = 32657300
   initE = 3375  #ParticipantTablighwhoPositive/totalscreeningat27/2/20
   initI = 1
   initR = 22
   initD = 0
   initS = initN - (initE + initI + initR + initD)
   beta= 12.5#15.44 
   sigma = 0.19
   gamma = 0.279 
   mu = 0.1

   days = 101

   params = Parameters()
   params.add('beta', value=beta, min=0, max=100)
   params.add('sigma', value=sigma, min=0, max=100)
   params.add('gamma', value=gamma, min=0, max=100)
   params.add('mu', value=mu, min=0, max=100)

   initial_conditions = [initS,initE, initI, initR, initD]
   params['beta'].value, params['sigma'].value,params['gamma'].value, params['mu'].value = [beta, sigma, gamma, mu]
   tspan = np.arange(0, days, 1) #timespan,np.arrange to arrange day 0 till days with increment of 1
   sol = ode_solver(tspan, initial_conditions, params)
   S, E, I, R, D = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4]


   fig = go.Figure()

   fig.add_trace(go.Scatter(x=tspan, y=S, mode='lines',line_color='blue', name='Susceptible'))
   fig.add_trace(go.Scatter(x=tspan, y=E, mode='lines',line_color='turquoise', name='Exposed'))
   fig.add_trace(go.Scatter(x=tspan, y=I, mode='lines', line_color='purple', name='Infected'))
   fig.add_trace(go.Scatter(x=tspan, y=R, mode='lines', line_color='orange',name='Recovered'))
   fig.add_trace(go.Scatter(x= tspan, y=D, mode='lines', line_color='red',name='Death'))
    

   if days <= 30:
    step = 1
   elif days <= 90:
    step = 7 
   else:
    step = 10


   # Edit the layout
   fig.update_layout(title='Simulation of Classical SEIRD Model',
            xaxis_title='Days',
            yaxis_title='Populations',
            title_x=0.5, font_size= 22,
            width=1000, height=600
                     )
   fig.update_xaxes(tickangle=0, tickformat = None ,tickmode='array', tickvals=np.arange(0, days + 1,step))


   fig.write_image("images/Cseird_simulation_withoutHD_non-optimised_constantparams.png")
   fig.show()

   #suggestion: better to change days to date similar like plotting figures for historical data
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

