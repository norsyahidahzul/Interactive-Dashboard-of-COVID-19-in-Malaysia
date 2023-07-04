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
        ("cSEIRD1", "cSEIRD2", "mSEIRD1", "mSEIRD2")
    )

 if model == 'cSEIRD1':
   populations = ["All populations","Susceptible","Exposed","Infected","Recovered","Death"]
   instructions = """
    Choose any populations to display\n
    Options: either all populations or every each of populations\n
    """
   #selectbox for populations
   selected_populations = st.sidebar.selectbox("Choose populations to display", populations, help=instructions,)
   
   days=st.sidebar.slider("Choose a value of Time-window (days)", min_value=0.0, max_value=100.0, value=120.0, step=1.0)
   beta=st.sidebar.slider("Choose a value of Infection rate (β)", min_value=0.0, max_value=100.0, value=1.09, step=0.1)
   sigma=st.sidebar.slider("Choose a value of Incubation rate (σ)", min_value=0.0, max_value=100.0, value=0.19, step=0.1)
   gamma=st.sidebar.slider("Choose a value of Recovery rate (γ)", min_value=0.0, max_value=100.0, value=0.28, step=0.1)
   mu=st.sidebar.slider("Choose a value of Death rate (μ)", min_value=0.0, max_value=100.0, value=0.1, step=0.1)

   st.write('**You have selected Classical SEIRD model with constant parameters simulator :smile:.**')
   st.subheader("**cSEIRD1: Classical SEIRD model with constant parameters simulator**")
   
  
   if st.button('Click here for more info on its mathematical expressions'):
         st.write('The mathematical expression of classical SEIRD model with constant parameters can be expressed in terms of ordinary differential equations (ODE) as follows as follows:')
         st.latex(r'''\frac{dS}{dt} = -\frac{\beta SI}{N}''') #left(t\right)
         st.latex(r'''\frac{dE}{dt} = \frac{\beta SI}{N} ''')
         st.latex(r'''\frac{dI}{dt} = \sigma E - \gamma I- \mu I''')
         st.latex(r'''\frac{dR}{dt} =  \gamma I''')
         st.latex(r'''\frac{dD}{dt} =  \mu I''')
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
     
   fig.update_layout(title='Simulation of Classical SEIRD Model with Constant Parameters',
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
 if model == 'mSEIRD1':
   #selectbox for populations
   selected_populations = st.sidebar.selectbox('Choose populations to display', ['All populations','Susceptible','Exposed','Infected','Recovered','Death'], help=instructions)
   days=st.sidebar.slider("Choose a value of Time-window (days)", min_value=0.0, max_value=100.0, value=120.0, step=1.0)
   beta_I=st.sidebar.slider("Choose a value of Infection rate (βI)", min_value=0.0, max_value=100.0, value=1.09, step=0.1)
   beta_E=st.sidebar.slider("Choose a value of Infection rate (βE)", min_value=0.0, max_value=200.0, value=1.09, step=0.1)
   sigma=st.sidebar.slider("Choose a value of Incubation rate (σ)", min_value=0.0, max_value=100.0, value=0.19, step=0.1)
   gamma=st.sidebar.slider("Choose a value of Recovery rate (γ)", min_value=0.0, max_value=100.0, value=0.28, step=0.1)
   mu=st.sidebar.slider("Choose a value of Death rate (μ)", min_value=0.0, max_value=100.0, value=0.02, step=0.01)
   delta=st.sidebar.slider("Choose a value of Reinfection rate (δ)", min_value=0.0, max_value=100.0, value=0.2, step=0.1)
   
   st.write('**You have selected Modified SEIRD model simulator with constant parameters :smile:.**')
   st.subheader("**Modified SEIRD model with constant parameters simulator**")
   
   if st.button('Click here for more info on its mathematical expressions'):
         st.write('The mathematical expression of modified SEIRD model with constant parameters can be expressed in terms of ordinary differential equations (ODE) as follows:')
         st.latex(r'''\frac{ds}{dt} = -\frac{\beta_I SI}{N} -\frac{\beta_E SE}{N} 
         + \delta R''')
         st.latex(r'''\frac{dE}{dt} = \frac{\beta_I SI}{N} +\frac{\beta_E SE}{N} 
          - \sigma E''')
         st.latex(r'''\frac{dI}{dt} = \sigma E - \gamma I- \mu I''')
         st.latex(r'''\frac{dR}{dt} =  \gamma I- \delta R''')
         st.latex(r'''\frac{dD}{dt} =  \mu I''')
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
     
   fig.update_layout(title='Simulation of Modified SEIRD Model with Constant Parameters',
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

 if model == 'cSEIRD2':
   #selectbox for populations
   selected_populations = st.sidebar.selectbox('Choose populations to display', ['All populations','Susceptible','Exposed','Infected','Recovered','Death'], help=instructions)
   
   days=st.sidebar.slider("Choose a value of time-window (days)", min_value=0.0, max_value=100.0, value=120.0, step=1.0)
   beta0=st.sidebar.slider("Choose a value of infection rate at first day of MCO(β0)", min_value=0.0001, max_value=1.0, value=0.1610073221, step=0.1)
   beta1=st.sidebar.slider("Choose a value of infection rate expected gradient (β1)", min_value=0.0001, max_value=1.0, value= 0.001423470338643, step=0.1)
   beta2=st.sidebar.slider("Choose a value of infection rate at first day of pre-MCO(β2)", min_value=0.0001, max_value=1.0, value=0.0737333545187117, step=0.1)
   cha_time_beta=st.sidebar.slider("Choose a value of characteristic time of infection rate(τβ)", min_value=0.0001, max_value=100.0, value=21.732155377825364, step=0.1)
   sigma=st.sidebar.slider("Choose a value of incubation rate (σ)", min_value=0.0, max_value=1.0, value=0.15, step=0.1)
   gamma0=st.sidebar.slider("Choose a value of recovery rate at first day of MCO(γ0)", min_value=0.0001, max_value=1.0, value=0.025909834357211, step=0.1)
   gamma1=st.sidebar.slider("Choose a value of recovery rate supremum value(γ1)", min_value=0.0001, max_value=1.0, value=0.026700039069596, step=0.1)
   gamma2=st.sidebar.slider("Choose a value of recovery rate expected gradient(γ2)", min_value=0.00001, max_value=1.0, value=0.00008300, step=0.1)
   gamma3=st.sidebar.slider("Choose a value of recovery rate at first day of pre-MCO(γ3)", min_value=0.0001, max_value=1.0, value=0.006066878694908, step=0.1)
   cha_time_gamma=st.sidebar.slider("Choose a value of characteristic time of recovery rate(τγ)", min_value=0.0, max_value=100.0, value=12.359300607126448, step=0.1)
   mu0=st.sidebar.slider("Choose a value of death rate at first day of MCO (μ0)", min_value=0.001, max_value=1.0, value=0.001510617364145, step=0.01)
   mu1=st.sidebar.slider("Choose a value of death rate at infinite time, last day of simulation(μ1)", min_value=0.0, max_value=1.0, value=0.000153164803984, step=0.01)
   mu2=st.sidebar.slider("Choose a value of death rate expected gradient (μ2)", min_value=0.0000001, max_value=1.0, value=0.000080126119201, step=0.01)
   mu3=st.sidebar.slider("Choose a value of death rate at first day of pre-MCO (μ3)", min_value=0.00001, max_value=1.0, value=0.000250643745785, step=0.01)
   cha_time_mu=st.sidebar.slider("Choose a value of characteristic time of death rate(τμ)", min_value=0.0, max_value=100.0, value=26.359322685088163, step=0.1)
   delta=st.sidebar.slider("Choose a value of Reinfection rate (δ)", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
   p=st.sidebar.slider("Choose a value of infection rates ratio (p)", min_value=1.0, max_value=5.0, value=1.0, step=0.1)
   r=st.sidebar.slider("Choose a value of quarantine rule-abiding population proportion(r)", min_value=1.0, max_value=100.0, value=1.0, step=0.1)
   tMCO=st.sidebar.slider("Choose a value of MCO first day", min_value=1.0, max_value=100.0, value=20.0, step=0.1)
   tRMCO=st.sidebar.slider("Choose a value of MCO lifted-up day", min_value=1.0, max_value=200.0, value=140.0, step=0.1)
   
   st.write('**You have selected Classical SEIRD model simulator with time-varying parameters :smile:.**')
   st.subheader("**Classical SEIRD model with time-varying parameters simulator**")
   
   if st.button('Click here for more info on its mathematical expressions'):
         st.write('The mathematical expression of modified SEIRD model in terms of ordinary differential equations (ODE) can be expressed as follows:')
         st.latex(r'''\frac{ds}{dt} = -\frac{\beta\left(t\right)SI}{N}  
         ''')
         st.latex(r'''\frac{dE}{dt} = \frac{\beta_I\left(t\right)SI}{N} 
          - \sigma E''')
         st.latex(r'''\frac{dI}{dt} = \sigma E - \gamma\left(t\right)I- \mu\left(t\right)I''')
         st.latex(r'''\frac{dR}{dt} =  \gamma\left(t\right)I''')
         st.latex(r'''\frac{dD}{dt} =  \mu\left(t\right)I''')    
         st.write('___________________________________________________________________________________________________________') 
         st.write('Time-varying parameters βI(t), βE(t), γ(t) and μ(t) are formulated in piecewise-defined function as follows:')  
         #insert the formula
   st.write('___________________________________________________________________________________________________________')    
   
 

   #define piecewise defined funtion for Beta_I and Beta_E, gamma,mu
   def beta(tspan):
        if tspan < tMCO:
            return  (beta1*tspan) + beta2
        elif tMCO<=tspan <tRMCO:
            return beta0*np.exp(-((tspan-tMCO)/(cha_time_beta)))
        else:
            return (1-r)*(beta1*(tspan-tRMCO) + beta2)


   def gamma (tspan):
        if tspan < tMCO:
            return gamma2*tspan+gamma3
        else:
            return gamma0+((gamma1)/(1+np.exp(-tspan+tMCO+cha_time_gamma)))

   def mu(tspan):
        if tspan < tMCO:
            return mu2*tspan+mu3
        elif tMCO<=tspan<tRMCO:
           return mu0*np.exp(-(tspan-tMCO)/cha_time_mu)+mu1
        else:
            return  mu2*(tspan-tRMCO)+ mu3
   
   #define Model     
   def ode_model(z, tspan, beta0, beta1, beta2, cha_time_beta,gamma0,gamma1,gamma2,gamma3,mu0,mu1,mu2,mu3,cha_time_mu, cha_time_gamma,sigma,r):
        S, E, I, R, D = z
        N = S + E + I + R + D
        dSdt = -(beta(tspan)*S*I)/N 
        dEdt = (beta(tspan)*S*I)/N  - sigma*E
        dIdt = sigma*E - gamma(tspan)*I - mu(tspan)*I
        dRdt = gamma(tspan)*I 
        dDdt = mu(tspan)*I
        return [dSdt, dEdt, dIdt, dRdt, dDdt]
        
    #define ODE Solver
   def ode_solver(t, initial_conditions, params):
        initS, initE, initI, initR, initD = initial_conditions
        #beta, sigma, gamma, mu= params['beta'].value, params['sigma'].value, params['gamma'].value, params['mu'].value
        res = odeint(ode_model, [initS, initE, initI, initR, initD], t, args=(beta0, beta1, beta2, cha_time_beta,gamma0,gamma1,gamma2,gamma3,mu0,mu1,mu2,mu3,cha_time_mu, cha_time_gamma,sigma,r)) 
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
     
   fig.update_layout(title='Simulation of Classical SEIRD Model with Time-Varying Parameters',
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
   if beta (days) >= 3.58:
    st.subheader ('Warning!')
    st.write("""Infection rate at selected days is too high and R0 is higher than 1.""")
    st.write("""**Recommendations**.""")
    st.write("""Government needs to:""")
    st.write("""1. Implement movement control order""")
    st.write("""2. Create awareness on importance of population behaviour towards pandemic
    """)
   else:
    st.subheader ('**Good job!**')
    st.write("""Infection rate at selected days is in a moderate value and R0 is lesser than 1. """)
    
 if model == 'mSEIRD2':
   #selectbox for populations
   selected_populations = st.sidebar.selectbox('Choose populations to display', ['All populations','Susceptible','Exposed','Infected','Recovered','Death'], help=instructions)
   
   days=st.sidebar.slider("Choose a value of time-window (days)", min_value=0.0, max_value=100.0, value=120.0, step=1.0)
   beta0=st.sidebar.slider("Choose a value of infection rate at first day of MCO(β0)", min_value=0.0001, max_value=1.0, value=0.1610073221, step=0.1)
   beta1=st.sidebar.slider("Choose a value of infection rate expected gradient (β1)", min_value=0.0001, max_value=1.0, value= 0.001423470338643, step=0.1)
   beta2=st.sidebar.slider("Choose a value of infection rate at first day of pre-MCO(β2)", min_value=0.0001, max_value=1.0, value=0.0737333545187117, step=0.1)
   cha_time_beta=st.sidebar.slider("Choose a value of characteristic time of infection rate(τβ)", min_value=0.0001, max_value=100.0, value=21.732155377825364, step=0.1)
   sigma=st.sidebar.slider("Choose a value of incubation rate (σ)", min_value=0.0, max_value=1.0, value=0.15, step=0.1)
   gamma0=st.sidebar.slider("Choose a value of recovery rate at first day of MCO(γ0)", min_value=0.0001, max_value=1.0, value=0.025909834357211, step=0.1)
   gamma1=st.sidebar.slider("Choose a value of recovery rate supremum value(γ1)", min_value=0.0001, max_value=1.0, value=0.026700039069596, step=0.1)
   gamma2=st.sidebar.slider("Choose a value of recovery rate expected gradient(γ2)", min_value=0.00001, max_value=1.0, value=0.00008300, step=0.1)
   gamma3=st.sidebar.slider("Choose a value of recovery rate at first day of pre-MCO(γ3)", min_value=0.0001, max_value=1.0, value=0.006066878694908, step=0.1)
   cha_time_gamma=st.sidebar.slider("Choose a value of characteristic time of recovery rate(τγ)", min_value=0.0, max_value=100.0, value=12.359300607126448, step=0.1)
   mu0=st.sidebar.slider("Choose a value of death rate at first day of MCO (μ0)", min_value=0.001, max_value=1.0, value=0.001510617364145, step=0.01)
   mu1=st.sidebar.slider("Choose a value of death rate at infinite time, last day of simulation(μ1)", min_value=0.0, max_value=1.0, value=0.000153164803984, step=0.01)
   mu2=st.sidebar.slider("Choose a value of death rate expected gradient (μ2)", min_value=0.0000001, max_value=1.0, value=0.000080126119201, step=0.01)
   mu3=st.sidebar.slider("Choose a value of death rate at first day of pre-MCO (μ3)", min_value=0.00001, max_value=1.0, value=0.000250643745785, step=0.01)
   cha_time_mu=st.sidebar.slider("Choose a value of characteristic time of death rate(τμ)", min_value=0.0, max_value=100.0, value=26.359322685088163, step=0.1)
   delta=st.sidebar.slider("Choose a value of Reinfection rate (δ)", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
   p=st.sidebar.slider("Choose a value of infection rates ratio (p)", min_value=1.0, max_value=5.0, value=1.0, step=0.1)
   r=st.sidebar.slider("Choose a value of quarantine rule-abiding population proportion(r)", min_value=1.0, max_value=100.0, value=1.0, step=0.1)
   tMCO=st.sidebar.slider("Choose a value of MCO first day", min_value=1.0, max_value=100.0, value=20.0, step=0.1)
   tRMCO=st.sidebar.slider("Choose a value of MCO lifted-up day", min_value=1.0, max_value=100.0, value=200.0, step=0.1)
   
   st.write('**You have selected Modified SEIRD model simulator with time-varying parameters :smile:.**')
   st.subheader("**Modified SEIRD model with time-varying parameters simulator**")
   
   if st.button('Click here for more info on its mathematical expressions'):
         st.write('The mathematical expression of modified SEIRD model in terms of ordinary differential equations (ODE) can be expressed as follows:')
         st.latex(r'''\frac{ds}{dt} = -\frac{\beta_I\left(t\right)SI}{N} -\frac{\beta_E\left(t\right)SE}{N} 
         + \delta\left(t\right)R''')
         st.latex(r'''\frac{dE}{dt} = \frac{\beta_I\left(t\right)SI}{N} +\frac{\beta_E\left(t\right)SE}{N} 
          - \sigma E''')
         st.latex(r'''\frac{dI}{dt} = \sigma E - \gamma\left(t\right)I- \mu\left(t\right)I''')
         st.latex(r'''\frac{dR}{dt} =  \gamma\left(t\right)I- \delta\left(t\right)R''')
         st.latex(r'''\frac{dD}{dt} =  \mu\left(t\right)I''')    
         st.write('___________________________________________________________________________________________________________') 
         st.write('Time-varying parameters βI(t), βE(t), γ(t) and μ(t) are formulated in piecewise-defined function as follows:')  
         #insert the formula
   st.write('___________________________________________________________________________________________________________')    
   
 

   #define piecewise defined funtion for Beta_I and Beta_E, gamma,mu
   def beta_I(tspan):
        if tspan < tMCO:
            return  (beta1*tspan) + beta2
        elif tMCO<=tspan <tRMCO:
            return beta0*np.exp(-((tspan-tMCO)/(cha_time_beta)))
        else:
            return (1-r)*(beta1*(tspan-tRMCO) + beta2)

   def beta_E(tspan):
        if tspan < tMCO:
            return p*(beta1*tspan+beta2)
        elif tMCO<=tspan<tRMCO:
           return p*(beta0*np.exp(-((tspan-tMCO)/(cha_time_beta))))
        else:
            return  p*((1-r)*(beta1*(tspan-tRMCO)+beta2))

   def gamma (tspan):
        if tspan < tMCO:
            return gamma2*tspan+gamma3
        else:
            return gamma0+((gamma1)/(1+np.exp(-tspan+tMCO+cha_time_gamma)))

   def mu(tspan):
        if tspan < tMCO:
            return mu2*tspan+mu3
        elif tMCO<=tspan<tRMCO:
           return mu0*np.exp(-(tspan-tMCO)/cha_time_mu)+mu1
        else:
            return  mu2*(tspan-tRMCO)+ mu3
   
   #define Model     
   def ode_model(z, tspan, beta0, beta1, beta2, cha_time_beta,gamma0,gamma1,gamma2,gamma3,mu0,mu1,mu2,mu3,cha_time_mu, cha_time_gamma,sigma,delta,p,r):
        S, E, I, R, D = z
        N = S + E + I + R + D
        dSdt = -(beta_I(tspan)*S*I)/N - (beta_E(tspan)*S*E)/N + delta*R
        dEdt = (beta_I(tspan)*S*I)/N + (beta_E(tspan)*S*E)/N - sigma*E
        dIdt = sigma*E - gamma(tspan)*I - mu(tspan)*I
        dRdt = gamma(tspan)*I - delta*R
        dDdt = mu(tspan)*I
        return [dSdt, dEdt, dIdt, dRdt, dDdt]
        
    #define ODE Solver
   def ode_solver(t, initial_conditions, params):
        initS, initE, initI, initR, initD = initial_conditions
        #beta, sigma, gamma, mu= params['beta'].value, params['sigma'].value, params['gamma'].value, params['mu'].value
        res = odeint(ode_model, [initS, initE, initI, initR, initD], t, args=(beta0, beta1, beta2, cha_time_beta,gamma0,gamma1,gamma2,gamma3,mu0,mu1,mu2,mu3,cha_time_mu, cha_time_gamma,sigma,delta,p,r)) 
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
     
   fig.update_layout(title='Simulation of Modified SEIRD Model with Time-Varying Parameters',
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
   if beta_I (days) >= 3.58:
    st.subheader ('Warning!')
    st.write("""Infection rate at selected days is too high and R0 is higher than 1.""")
    st.write("""**Recommendations**.""")
    st.write("""Government needs to:""")
    st.write("""1. Implement movement control order""")
    st.write("""2. Create awareness on importance of population behaviour towards pandemic
    """)
   else:
    st.subheader ('**Good job!**')
    st.write("""Infection rate at selected days is in a moderate value and R0 is lesser than 1. """)
    
st.set_page_config(page_title="COVID-19 Simulator", page_icon="📈",layout="wide")
st.markdown("# COVID-19 Simulator")
st.sidebar.header("COVID-19 Simulator")
st.write(
    """This simulator shows the effects of considered factors upon its increment or decrement in values. You may try to choose 
    the values of parameters by moving the slider at the sidebar. See the changes of the SEIRD simulation and 
    how the value of epidemiological parameters affects COVID-19 cases. Enjoy!"""
)

COVID19_Simulator()
    
    
   
