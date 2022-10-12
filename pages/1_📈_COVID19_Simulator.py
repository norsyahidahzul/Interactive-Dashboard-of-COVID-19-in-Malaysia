import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from lmfit import Parameters
import plotly.graph_objects as go



def COVID19_Simulator(): 
   
## sidebar
# Using "with" notation
 with st.sidebar:
    model = st.radio(
        "Choose a type of SEIRD model simulator",
        ("Classical", "Modified")
    )

 if model == 'Classical':
   populations = ["All","Susceptible","Exposed","Infected","Recovered","Death"]
   instructions = """
    Choose any populations to display\n
    Options: either all populations or every each of populations\n
    """
   #selectbox for populations
   selected_populations = st.sidebar.selectbox("Choose populations to display", populations, help=instructions,)
   
   days=st.sidebar.slider("Choose a value of Time-window (days)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
   beta=st.sidebar.slider("Choose a value of Infection rate (β)", min_value=0.0, max_value=100.0, value=15.44, step=0.1)
   sigma=st.sidebar.slider("Choose a value of Incubation rate (σ)", min_value=0.0, max_value=100.0, value=0.19, step=0.1)
   gamma=st.sidebar.slider("Choose a value of Recovery rate (γ)", min_value=0.0, max_value=100.0, value=0.28, step=0.1)
   mu=st.sidebar.slider("Choose a value of Death rate (μ)", min_value=0.0, max_value=100.0, value=0.1, step=0.1)

   st.write('**You have selected Classical SEIRD model simulator :smile:.**')
   st.subheader("**Classical SEIRD model simulator**")
   
  
   if st.button('Click here for more info on its mathematical expressions'):
         st.write('The mathematical expression of classical SEIRD model in terms of ordinary differential equations (ODE) can be expressed as follows:')
         st.latex(r'''\frac{ds}{dt} = -\frac{\beta S\left(t\right)I\left(t\right)}{N}''')
         st.latex(r'''\frac{dE}{dt} = \frac{\beta S\left(t\right)I\left(t\right)}{N} ''')
         st.latex(r'''\frac{dI}{dt} = \sigma E\left(t\right) - \gamma I\left(t\right)- \mu I\left(t\right)''')
         st.latex(r'''\frac{dR}{dt} =  \gamma I\left(t\right)''')
         st.latex(r'''\frac{dD}{dt} =  \delta R\left(t\right)''')
   st.write('___________________________________________________________________________________________________________')   
  
    
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
        #beta, sigma, gamma, mu= params['beta'].value, params['sigma'].value, params['gamma'].value, params['mu'].value
        res = odeint(ode_model, [initS, initE, initI, initR, initD], t, args=(beta, sigma, gamma, mu)) 
        return res    
        
    #initial condition and initial values of parameters
    #initN (Malaysian Population 2020- include non citizen)
   initN = 32657300
   initE = 3375  #ParticipantTablighwhoPositive/totalscreeningat27/2/20
   initI = 1
   initR = 22
   initD = 0
   initS = initN - (initE + initI + initR + initD)
   params = Parameters()

   initial_conditions = [initS,initE, initI, initR, initD]
   #params['beta'].value, params['sigma'].value,params['gamma'].value, params['mu'].value = [beta, sigma, gamma, mu]
   tspan = np.arange(0, days, 1) #timespan,np.arrange to arrange day 0 till days with increment of 1
   sol = ode_solver(tspan, initial_conditions, params)
   S, E, I, R, D = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4]

  
   fig = go.Figure()
   #to able functionality of selectbox, put if statement
   if selected_populations == 'Susceptible':
    fig.add_trace(go.Scatter(x=tspan, y=S, mode='lines',line_color='blue', name='Susceptible'))
   elif selected_populations == 'Exposed':
    fig.add_trace(go.Scatter(x=tspan, y=E, mode='lines',line_color='turquoise', name='Exposed'))
   elif selected_populations == 'Infected':
    fig.add_trace(go.Scatter(x=tspan, y=I, mode='lines',line_color='purple', name='Infected'))
   elif selected_populations == 'Recovered':
    fig.add_trace(go.Scatter(x=tspan, y=R, mode='lines',line_color='orange', name='Recovered'))
   elif selected_populations == 'Death':
    fig.add_trace(go.Scatter(x=tspan, y=D, mode='lines',line_color='red', name='Death'))
   
   
   else:
    fig.add_trace(go.Scatter(x=tspan, y=S, mode='lines',line_color='blue', name='Susceptible'))
    fig.add_trace(go.Scatter(x=tspan, y=E, mode='lines',line_color='turquoise', name='Exposed'))
    fig.add_trace(go.Scatter(x=tspan, y=I, mode='lines', line_color='purple', name='Infected'))
    fig.add_trace(go.Scatter(x=tspan, y=R, mode='lines', line_color='orange',name='Recovered'))
    fig.add_trace(go.Scatter(x= tspan, y=D, mode='lines', line_color='red',name='Death'))
   
   if days <= 30:
     step = 2
   elif days <= 90:
     step = 7 
   else:
     step = 10
     
   fig.update_layout(title='Simulation of Classical SEIRD Model',
            xaxis_title='Days',
            yaxis_title='Populations',
            title_x=0.5, font_size= 22,
            width=1000, height=600
                     )
   fig.update_xaxes(tickangle=0, tickformat = None ,tickmode='array', tickvals=np.arange(0, days + 1,step))
   st.write(fig) 
   #st.pyplot(fig)        
   
   st.write('You have selected:',selected_populations)
   
   st.write('___________________________________________________________________________________________________________') 
   if beta >= 3.58:
    st.subheader ('Warning!')
    st.write("""Infection rate is too high and R0 is higher than 1.""")
    st.write("""**Recommendations**.""")
    st.write("""Government needs to:""")
    st.write("""1. Implement movement control order""")
    st.write("""2. Create awareness on importance of population behaviour towards pandemic
    """)
   else:
    st.subheader ('**Good job!**')
    st.write("""Infection rate is in a moderate value and R0 lesser than 1. """)
   
 else:
   instructions = """
    Choose any populations to display\n
    Options: either all populations or every each of populations\n
    """
   
   #selectbox for populations
   
   selected_populations = st.sidebar.selectbox('Choose populations to display', ['All','Susceptible','Exposed','Infected','Recovered','Death'], help=instructions)
   days=st.sidebar.slider("Choose a value of Time-window (days)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
   beta_I=st.sidebar.slider("Choose a value of Infection rate (βI)", min_value=0.0, max_value=100.0, value=15.44, step=0.1)
   beta_E=st.sidebar.slider("Choose a value of Infection rate (βE)", min_value=0.0, max_value=200.0, value=77.2, step=0.1)
   sigma=st.sidebar.slider("Choose a value of Incubation rate (σ)", min_value=0.0, max_value=100.0, value=0.19, step=0.1)
   gamma=st.sidebar.slider("Choose a value of Recovery rate (γ)", min_value=0.0, max_value=100.0, value=0.28, step=0.1)
   mu=st.sidebar.slider("Choose a value of Death rate (μ)", min_value=0.0, max_value=100.0, value=0.02, step=0.01)
   delta=st.sidebar.slider("Choose a value of Reinfection rate (δ)", min_value=0.0, max_value=100.0, value=0.2, step=0.1)
   
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
   st.write('___________________________________________________________________________________________________________')    
   
   #define Model     
   def ode_model(z, t, beta_I,beta_E, sigma, gamma, mu,delta):
        S, E, I, R, D = z
        N = S + E + I + R + D
        dSdt = -beta_I*S*I/N-beta_E*S*I/N+delta*R
        dEdt = beta_I*S*I/N+beta_E*S*I/N - sigma*E
        dIdt = sigma*E - gamma*I - mu*I
        dRdt = gamma*I- delta*R
        dDdt = mu*I
        return [dSdt, dEdt, dIdt, dRdt, dDdt]
        
    #define ODE Solver
   def ode_solver(t, initial_conditions, params):
        initS, initE, initI, initR, initD = initial_conditions
        #beta, sigma, gamma, mu= params['beta'].value, params['sigma'].value, params['gamma'].value, params['mu'].value
        res = odeint(ode_model, [initS, initE, initI, initR, initD], t, args=(beta_I, beta_E, sigma, gamma, mu,delta)) 
        return res    
        
    #initial condition and initial values of parameters
    #initN (Malaysian Population 2020- include non citizen)
   initN = 32657300
   initE = 3375  #ParticipantTablighwhoPositive/totalscreeningat27/2/20
   initI = 1
   initR = 22
   initD = 0
   initS = initN - (initE + initI + initR + initD)
   params = Parameters()

   initial_conditions = [initS,initE, initI, initR, initD]
   #params['beta'].value, params['sigma'].value,params['gamma'].value, params['mu'].value = [beta, sigma, gamma, mu]
   tspan = np.arange(0, days, 1) #timespan,np.arrange to arrange day 0 till days with increment of 1
   sol = ode_solver(tspan, initial_conditions, params)
   S, E, I, R, D = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4]


   fig = go.Figure()
   #to able functionality of selectbox, put if statement
   fig = go.Figure()
   #to able functionality of selectbox, put if statement
   if selected_populations == 'Susceptible':
    fig.add_trace(go.Scatter(x=tspan, y=S, mode='lines',line_color='blue', name='Susceptible'))
   elif selected_populations == 'Exposed':
    fig.add_trace(go.Scatter(x=tspan, y=E, mode='lines',line_color='turquoise', name='Exposed'))
   elif selected_populations == 'Infected':
    fig.add_trace(go.Scatter(x=tspan, y=I, mode='lines',line_color='purple', name='Infected'))
   elif selected_populations == 'Recovered':
    fig.add_trace(go.Scatter(x=tspan, y=R, mode='lines',line_color='orange', name='Recovered'))
   elif selected_populations == 'Death':
    fig.add_trace(go.Scatter(x=tspan, y=D, mode='lines',line_color='red', name='Death'))
   
   
   else:
    fig.add_trace(go.Scatter(x=tspan, y=S, mode='lines',line_color='blue', name='Susceptible'))
    fig.add_trace(go.Scatter(x=tspan, y=E, mode='lines',line_color='turquoise', name='Exposed'))
    fig.add_trace(go.Scatter(x=tspan, y=I, mode='lines', line_color='purple', name='Infected'))
    fig.add_trace(go.Scatter(x=tspan, y=R, mode='lines', line_color='orange',name='Recovered'))
    fig.add_trace(go.Scatter(x= tspan, y=D, mode='lines', line_color='red',name='Death'))
   
   if days <= 30:
     step = 2
   elif days <= 90:
     step = 7 
   else:
     step = 10
     
   fig.update_layout(title='Simulation of Modified SEIRD Model',
            xaxis_title='Days',
            yaxis_title='Populations',
            title_x=0.5, font_size= 22,
            width=1000, height=600
                     )
   fig.update_xaxes(tickangle=0, tickformat = None ,tickmode='array', tickvals=np.arange(0, days + 1,step))
   st.write(fig) 
   #st.pyplot(fig)        
   
   
   st.write('You have selected:',selected_populations)
   st.write('___________________________________________________________________________________________________________') 
   if beta_I >= 3.58:
    st.subheader ('Warning!')
    st.write("""Infection rate is too high and R0 is higher than 1.""")
    st.write("""**Recommendations**.""")
    st.write("""Government needs to:""")
    st.write("""1. Implement movement control order""")
    st.write("""2. Create awareness on importance of population behaviour towards pandemic
    """)
   else:
    st.subheader ('**Good job!**')
    st.write("""Infection rate is in a moderate value and R0 is lesser than 1. """)

st.set_page_config(page_title="COVID-19 Simulator", page_icon="📈",layout="wide")
st.markdown("# COVID-19 Simulator")
st.sidebar.header("COVID-19 Simulator")
st.write(
    """This simulator shows the effects of considered factors upon its increment or decrement in values. You may try to choose 
    the values of parameters by moving the slider at the sidebar. See the changes of the SEIRD simulation and 
    how the value of epidemiological parameters affects COVID-19 cases. Enjoy!"""
)

COVID19_Simulator()
    
    
   
