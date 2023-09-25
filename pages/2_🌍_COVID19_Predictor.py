import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from lmfit import Parameters
import plotly.graph_objects as go

def COVID19_Predictor():
 #with st.sidebar:
    
 instructions1 = """
    Dataset 1: 25/1/2020-18/9/2020\n
    Dataset 2: 27/2/2020-15/10/2020
    """    
    
 instructions2 = """   
    Nelder-Mead optimisation algorithm will be used for optimising time-varying parameters of SEIRD model\n
    Nelder-Mead: directive free simplex-based direct search technique for multidimensional unconstrained optimization\n
    
    """
 with st.container ():
    #container_size = "300px"
    col1, col2= st.columns(2)
    with col1:
        #selectbox for dataset
        selected_dataset = st.selectbox("Dataset", ["Dataset1","Dataset2"], help=instructions1)
         
    with col2:
        #selectbox for optimisisation
        selected_optimisation_algorithm = st.selectbox("Non-optimised/optimised", ["non-optimised","optimised"], help=instructions2)  
        
    col1, col2 = st.columns(2)
   
    with col1:
        tRMCO=st.slider("MCO lifting date", min_value=1, max_value=200, value=137, step=1)
     
    with col2:    
       r= st.slider("Proportion of quarantine rule-abiding population (%)", min_value=0.0, max_value=1.0, value=0.6, step=0.011)
 
 # Read in data from the Google Sheet.
 # Uses st.cache_data to only rerun when the query changes or after 10 min.
 #@st.cache_data(ttl=600)
 def load_data(sheets_url):
    csv_url = sheets_url.replace("/edit#gid=", "/export?format=csv&gid=")
    return pd.read_csv(csv_url)
 df1 = load_data(st.secrets["public_gsheets_url_DS1"])
 df2 = load_data(st.secrets["public_gsheets_url_DS2"])
 
 if selected_dataset == 'Dataset1':
    
    covid_history = df1 
    
    #initial condition and initial values of parameters
    #initN (Malaysian Population 2020- include non citizen)   
    initN = 32657300
    days = st.sidebar.slider("Choose time-window (days)", min_value=0, max_value=395, value=264, step=1)
    tMCO=53
    
    initE = 0#3375  #ParticipantTablighwhoPositive/totalscreeningat27/2/20
    initI = 3
    initR = 0
    initD = 0
    initS = initN - (initE + initI + initR + initD)                      
        
    if selected_optimisation_algorithm == 'non-optimised':
      #initial parameters                                           
            beta0=0.1610073221
            beta1=0.001423470338643
            beta2=  0.0737333545187117
            cha_time_beta= 21.732155377825364         
            gamma0=0.025909834357211
            gamma1=0.026700039069596
            gamma2=0.00008300
            gamma3=0.006066878694908
            cha_time_gamma= 12.359300607126448

            mu0=0.001510617364145 #0.07285434# #0.07278481
            mu1=0.000153164803984 #0.01222592# #0.01223569
            mu2=0.000080126119201#0.34390103# #0.34449327
            mu3= 0.000250643745785#0.02429897# #0.02447233
            cha_time_mu=26.359322685088163#5.31175157# #5.32643972

            sigma= 0.15#0.30001983#0.30001935#0.15#0.5##0.49999927#0.21982829#
            delta= 0#0.02
            p= 1#1#1.42728098##5 #1.42580770          
            
            
            #define piecewise defined funtion for Beta_I and Beta_E
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
                    
             #Define Model 
            def ode_model(z, tspan,beta0, beta1, beta2, cha_time_beta,gamma0,gamma1,gamma2,gamma3,mu0,mu1,mu2,mu3,cha_time_mu, cha_time_gamma,sigma,p):

                S, E, I, R, D = z
                N = S + E + I + R + D
                dSdt = -(beta_I(tspan)*S*I)/N - (beta_E(tspan)*S*E)/N + delta*R
                dEdt = (beta_I(tspan)*S*I)/N + (beta_E(tspan)*S*E)/N - sigma*E
                dIdt = sigma*E - gamma(tspan)*I - mu(tspan)*I
                dRdt = gamma(tspan)*I - delta*R
                dDdt = mu(tspan)*I
                return [dSdt, dEdt, dIdt, dRdt, dDdt]
                
            params = Parameters()     
            params.add('beta0', value=beta0, min=0, max=1)
            params.add('beta1', value=beta1, min=0, max=1)
            params.add('beta2', value=beta2, min=0, max=1)
            params.add('cha_time_beta', value=cha_time_beta, min=0, max=100)
            params.add('gamma0', value=gamma0, min=0, max=1)
            params.add('gamma1', value=gamma1, min=0, max=1)
            params.add('gamma2', value=gamma2, min=0, max=1)
            params.add('gamma3', value=gamma3, min=0, max=1)
            params.add('mu0', value=mu0, min=0, max=1)
            params.add('mu1', value=mu1, min=0, max=1)
            params.add('mu2', value=mu2, min=0, max=1)
            params.add('mu3', value=mu3, min=0, max=1)
            params.add('cha_time_gamma', value=cha_time_gamma, min=0, max=100)
            params.add('cha_time_mu', value=cha_time_mu, min=0, max=100)
            params.add('sigma', value=sigma,min=0.07, max=0.5)# min=0.07, max=0.5)
            params.add('p', value=p,min=1, max=5)# min=0.07, max=0.5)
                 
            #define ODE Solver
            def ode_solver(tspan, initial_conditions, params):
                initS, initE, initI, initR, initD = initial_conditions
                beta0, beta1, beta2, cha_time_beta,gamma0,gamma1,gamma2,gamma3,mu0,mu1,mu2,mu3,cha_time_mu,\
                cha_time_gamma,sigma, p = params['beta0'].value, params['beta1'].value, \
                params['beta2'].value,  params['cha_time_beta'].value, params['gamma0'].value, params['gamma1'].value, \
                params['gamma2'].value,  params['gamma3'].value, params['mu0'].value, params['mu1'].value, \
                params['mu2'].value,  params['mu3'].value, params['cha_time_mu'].value,  params['cha_time_gamma'].value,\
                params['sigma'].value, params['p'].value
                initS = initN - (initE + initI + initR + initD)
                res = odeint(ode_model, [initS, initE, initI, initR, initD], tspan,args=(beta0, beta1, beta2, cha_time_beta, gamma0,gamma1,gamma2,
                                                                     gamma3,mu0,mu1,mu2,mu3,cha_time_gamma, cha_time_mu,sigma,p))
                return res
           

            initial_conditions = [initS,initE, initI, initR, initD]
            #params['beta'].value, params['sigma'].value,params['gamma'].value, params['mu'].value = [beta, sigma, gamma, mu]
            tspan = np.arange(0, days+1, 1) #timespan,np.arrange to arrange day 0 till days with increment of 1
            sol = ode_solver(tspan, initial_conditions, params)
            S, E, I, R, D = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4]
       
            fig = go.Figure()
            #to able functionality of selectbox, put if statement
            fig.add_trace(go.Scatter(x=df1['date'][0:tRMCO], y=I[0:tRMCO], mode='lines',line_color='purple', name='Simulated Phase I and II'))
            fig.add_trace(go.Scatter(x=df1['date'][tRMCO:days+1], y=I[tRMCO:days+1], mode='lines',line_color='orange', name='Phase III'))
            fig.add_trace(go.Scatter(x=df1['date'], y=covid_history.iloc[0:days+1].daily_active_cases, mode='markers', marker_symbol='square',marker_color='purple',\
                                 name='Actual data', line = dict(dash='dash')))
        
         
            fig.update_layout(title='',
                xaxis_title='Days',
                yaxis_title='The number of active cases (I)',
                title_x=0.3, font_size= 22,
                width=700, height=500)
            fig.update_xaxes(tickangle=-45, tickformat = '%b %d',title_font=dict(size=20), tickfont=dict(size=20))
            fig.update_yaxes(title_font=dict(size=20), tickfont=dict(size=20))
            fig.update_layout(legend=dict(font=dict(size=20)))
            fig.update_layout(
            title='',
            margin=dict(t=0) # Set the top margin to 0
)
            st.write(fig) 
           
        
            #MAE and RMSE
            predicted_IRD = sol [:,2:5]
            observed_IRD = covid_history.loc[0:days, ['daily_active_cases','total_recovered_cases', 'total_death_cases', ]].values
            
            #st.markdown("##### Chosen time-windows: 25/1/2020-15/10/2020")                      
            text = "<i>Created by Norsyahidah Zulkarnain, Department of Computational & Theoretical Sciences, Kulliyyah of Science, IIUM</i>"         
            st.markdown(f"<span style='font-size:12px;'>{text}</span>", unsafe_allow_html=True)
        
                                                               
    if selected_optimisation_algorithm == 'optimised':
             #optimised parameters by NM
            
             #initial parameters                                           
            beta0=0.1610073221
            beta1=0.001423470338643
            beta2=  0.0737333545187117
            cha_time_beta= 21.732155377825364         
            gamma0=0.025909834357211
            gamma1=0.026700039069596
            gamma2=0.00008300
            gamma3=0.006066878694908
            cha_time_gamma= 12.359300607126448

            mu0=0.001510617364145 #0.07285434# #0.07278481
            mu1=0.000153164803984 #0.01222592# #0.01223569
            mu2=0.000080126119201#0.34390103# #0.34449327
            mu3= 0.000250643745785#0.02429897# #0.02447233
            cha_time_mu=26.359322685088163#5.31175157# #5.32643972

            sigma= 0.49948386		#0.30001983#0.30001935#0.15#0.5##0.49999927#0.21982829#
            delta= 0#0.02         
            p= 1.00000000#1#1.42728098##5 #1.42580770
          
            
            #define piecewise defined funtion for Beta_I and Beta_E
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
                    
             #Define Model 
            def ode_model(z, tspan,beta0, beta1, beta2, cha_time_beta,gamma0,gamma1,gamma2,gamma3,mu0,mu1,mu2,mu3,cha_time_mu, cha_time_gamma,sigma,p):

                S, E, I, R, D = z
                N = S + E + I + R + D
                dSdt = -(beta_I(tspan)*S*I)/N - (beta_E(tspan)*S*E)/N + delta*R
                dEdt = (beta_I(tspan)*S*I)/N + (beta_E(tspan)*S*E)/N - sigma*E
                dIdt = sigma*E - gamma(tspan)*I - mu(tspan)*I
                dRdt = gamma(tspan)*I - delta*R
                dDdt = mu(tspan)*I
                return [dSdt, dEdt, dIdt, dRdt, dDdt]
                
            params = Parameters()     
            params.add('beta0', value=beta0, min=0, max=1)
            params.add('beta1', value=beta1, min=0, max=1)
            params.add('beta2', value=beta2, min=0, max=1)
            params.add('cha_time_beta', value=cha_time_beta, min=0, max=100)
            params.add('gamma0', value=gamma0, min=0, max=1)
            params.add('gamma1', value=gamma1, min=0, max=1)
            params.add('gamma2', value=gamma2, min=0, max=1)
            params.add('gamma3', value=gamma3, min=0, max=1)
            params.add('mu0', value=mu0, min=0, max=1)
            params.add('mu1', value=mu1, min=0, max=1)
            params.add('mu2', value=mu2, min=0, max=1)
            params.add('mu3', value=mu3, min=0, max=1)
            params.add('cha_time_gamma', value=cha_time_gamma, min=0, max=100)
            params.add('cha_time_mu', value=cha_time_mu, min=0, max=100)
            params.add('sigma', value=sigma,min=0.07, max=0.5)# min=0.07, max=0.5)
            params.add('p', value=p,min=1, max=5)# min=0.07, max=0.5)
                 
            #define ODE Solver
            def ode_solver(tspan, initial_conditions, params):
                initS, initE, initI, initR, initD = initial_conditions
                beta0, beta1, beta2, cha_time_beta,gamma0,gamma1,gamma2,gamma3,mu0,mu1,mu2,mu3,cha_time_mu,\
                cha_time_gamma,sigma, p = params['beta0'].value, params['beta1'].value, \
                params['beta2'].value,  params['cha_time_beta'].value, params['gamma0'].value, params['gamma1'].value, \
                params['gamma2'].value,  params['gamma3'].value, params['mu0'].value, params['mu1'].value, \
                params['mu2'].value,  params['mu3'].value, params['cha_time_mu'].value,  params['cha_time_gamma'].value,\
                params['sigma'].value, params['p'].value
                initS = initN - (initE + initI + initR + initD)
                res = odeint(ode_model, [initS, initE, initI, initR, initD], tspan,args=(beta0, beta1, beta2, cha_time_beta, gamma0,gamma1,gamma2,
                                                                     gamma3,mu0,mu1,mu2,mu3,cha_time_gamma, cha_time_mu,sigma,p))
                return res

            initS = initN - (initE + initI + initR + initD)
            

            initial_conditions = [initS,initE, initI, initR, initD]
            #params['beta'].value, params['sigma'].value,params['gamma'].value, params['mu'].value = [beta, sigma, gamma, mu]
            tspan = np.arange(0, days+1, 1) #timespan,np.arrange to arrange day 0 till days with increment of 1
            sol = ode_solver(tspan, initial_conditions, params)
            S, E, I, R, D = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4]
       
            fig = go.Figure()
            #to able functionality of selectbox, put if statement
            fig.add_trace(go.Scatter(x=df1['date'][0:tRMCO], y=I[0:tRMCO], mode='lines',line_color='purple', name='Simulated'))
            fig.add_trace(go.Scatter(x=df1['date'][tRMCO:days+1], y=I[tRMCO:days+1], mode='lines',line_color='orange', name='Simulated'))
            fig.add_trace(go.Scatter(x=df1['date'], y=covid_history.iloc[0:days+1].daily_active_cases, mode='markers', marker_symbol='square',marker_color='purple',\
                                 name='Actual data', line = dict(dash='dash')))
        
         
            fig.update_layout(title='',
                xaxis_title='Days (year 2020)',
                yaxis_title='The number of active cases (I)',
                title_x=0.3, font_size= 22,
                width=700, height=500, xaxis_range=[df1['date'][0],df1['date'][days+1]]
                         )
            fig.update_xaxes(tickangle=-45, tickformat =  '%d<Br> %b <Br> ', tickmode='array',title_font=dict(size=18), tickfont=dict(size=16))
            fig.update_yaxes(title_font=dict(size=18), tickfont=dict(size=16))
            fig.update_layout(legend=dict(font=dict(size=18)))
            fig.update_layout(
            title='',
            margin=dict(t=0)) # Set the top margin to 0
            st.write(fig) 
            #st.markdown("##### Chosen time-windows: 25/1/2020-18/9/2020")
            text = "<i>Created by Norsyahidah Zulkarnain, Department of Computational & Theoretical Sciences, Kulliyyah of Science, IIUM</i>"         
            st.markdown(f"<span style='font-size:12px;'>{text}</span>", unsafe_allow_html=True)
    
            
            

   
 if selected_dataset == 'Dataset2':
    
    covid_history = df2    
    
    #initial condition and initial values of parameters
    #initN (Malaysian Population 2020- include non citizen)
    initN = 32657300
    days = st.sidebar.slider("Choose time-window (days)", min_value=0, max_value=395, value=264, step=1)
    tMCO=20.0    
    initE = 100#3375  #ParticipantTablighwhoPositive/totalscreeningat27/2/20
    initI = 1
    initR = 22
    initD = 0
    initS = initN - (initE + initI + initR + initD)
      
    if selected_optimisation_algorithm == 'non-optimised':
               
             #initial parameters                                           
            beta0=0.1610073221
            beta1=0.001423470338643
            beta2=  0.0737333545187117
            cha_time_beta= 21.732155377825364         
            gamma0=0.025909834357211
            gamma1=0.026700039069596
            gamma2=0.00008300
            gamma3=0.006066878694908
            cha_time_gamma= 12.359300607126448

            mu0=0.001510617364145 #0.07285434# #0.07278481
            mu1=0.000153164803984 #0.01222592# #0.01223569
            mu2=0.000080126119201#0.34390103# #0.34449327
            mu3= 0.000250643745785#0.02429897# #0.02447233
            cha_time_mu=26.359322685088163#5.31175157# #5.32643972

            sigma= 0.15#0.30001983#0.30001935#0.15#0.5##0.49999927#0.21982829#
            delta= 0#0.02
            p= 1#1#1.42728098##5 #1.42580770
            
            #define piecewise defined funtion for Beta_I and Beta_E
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
                    
             #Define Model 
            def ode_model(z, tspan,beta0, beta1, beta2, cha_time_beta,gamma0,gamma1,gamma2,gamma3,mu0,mu1,mu2,mu3,cha_time_mu, cha_time_gamma,sigma,p):

                S, E, I, R, D = z
                N = S + E + I + R + D
                dSdt = -(beta_I(tspan)*S*I)/N - (beta_E(tspan)*S*E)/N + delta*R
                dEdt = (beta_I(tspan)*S*I)/N + (beta_E(tspan)*S*E)/N - sigma*E
                dIdt = sigma*E - gamma(tspan)*I - mu(tspan)*I
                dRdt = gamma(tspan)*I - delta*R
                dDdt = mu(tspan)*I
                return [dSdt, dEdt, dIdt, dRdt, dDdt]
                
            params = Parameters()     
            params.add('beta0', value=beta0, min=0, max=1)
            params.add('beta1', value=beta1, min=0, max=1)
            params.add('beta2', value=beta2, min=0, max=1)
            params.add('cha_time_beta', value=cha_time_beta, min=0, max=100)
            params.add('gamma0', value=gamma0, min=0, max=1)
            params.add('gamma1', value=gamma1, min=0, max=1)
            params.add('gamma2', value=gamma2, min=0, max=1)
            params.add('gamma3', value=gamma3, min=0, max=1)
            params.add('mu0', value=mu0, min=0, max=1)
            params.add('mu1', value=mu1, min=0, max=1)
            params.add('mu2', value=mu2, min=0, max=1)
            params.add('mu3', value=mu3, min=0, max=1)
            params.add('cha_time_gamma', value=cha_time_gamma, min=0, max=100)
            params.add('cha_time_mu', value=cha_time_mu, min=0, max=100)
            params.add('sigma', value=sigma,min=0.07, max=0.5)# min=0.07, max=0.5)
            params.add('p', value=p,min=1, max=5)# min=0.07, max=0.5)
                 
            #define ODE Solver
            def ode_solver(tspan, initial_conditions, params):
                initS, initE, initI, initR, initD = initial_conditions
                beta0, beta1, beta2, cha_time_beta,gamma0,gamma1,gamma2,gamma3,mu0,mu1,mu2,mu3,cha_time_mu,\
                cha_time_gamma,sigma, p = params['beta0'].value, params['beta1'].value, \
                params['beta2'].value,  params['cha_time_beta'].value, params['gamma0'].value, params['gamma1'].value, \
                params['gamma2'].value,  params['gamma3'].value, params['mu0'].value, params['mu1'].value, \
                params['mu2'].value,  params['mu3'].value, params['cha_time_mu'].value,  params['cha_time_gamma'].value,\
                params['sigma'].value, params['p'].value
                initS = initN - (initE + initI + initR + initD)
                res = odeint(ode_model, [initS, initE, initI, initR, initD], tspan,args=(beta0, beta1, beta2, cha_time_beta, gamma0,gamma1,gamma2,
                                                                     gamma3,mu0,mu1,mu2,mu3,cha_time_gamma, cha_time_mu,sigma,p))
                return res
           

            initial_conditions = [initS,initE, initI, initR, initD]
            #params['beta'].value, params['sigma'].value,params['gamma'].value, params['mu'].value = [beta, sigma, gamma, mu]
            tspan = np.arange(0, days+1, 1) #timespan,np.arrange to arrange day 0 till days with increment of 1
            sol = ode_solver(tspan, initial_conditions, params)
            S, E, I, R, D = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4]
       
            fig = go.Figure()
            #to able functionality of selectbox, put if statement
            fig.add_trace(go.Scatter(x=df2['date'][0:tRMCO], y=I[0:tRMCO], mode='lines',line_color='purple', name='Simulated'))
            fig.add_trace(go.Scatter(x=df2['date'][tRMCO:days+1], y=I[tRMCO:days+1], mode='lines',line_color='orange', name='Simulated'))
            fig.add_trace(go.Scatter(x=df2['date'], y=covid_history.iloc[0:days+1].daily_active_cases, mode='markers', marker_symbol='square',marker_color='purple',\
                                 name='Actual data', line = dict(dash='dash')))
        
         
            fig.update_layout(title='',
                xaxis_title='Days (year 2020)',
                yaxis_title='The number of active cases (I)',
                title_x=0.3, font_size= 22,
                width=700, height=500, xaxis_range=[df2['date'][0],df2['date'][days+1]], yaxis_range=[0,4000]
                         )
            fig.update_xaxes(tickangle=-45 , tickformat =  '%d<Br> %b <Br> ', tickmode='array',title_font=dict(size=18), tickfont=dict(size=16))
            fig.update_yaxes(title_font=dict(size=18), tickfont=dict(size=16))
            fig.update_layout(legend=dict(font=dict(size=18)))
            fig.update_layout(
            title='',
            margin=dict(t=0))# Set the top margin to 0
            st.write(fig) 
            
            #st.markdown("##### Chosen time-windows: 27/2/2020-21/10/2020")
            text = "<i>Created by Norsyahidah Zulkarnain, Department of Computational & Theoretical Sciences, Kulliyyah of Science, IIUM</i>"         
            st.markdown(f"<span style='font-size:12px;'>{text}</span>", unsafe_allow_html=True)
                             
               
                                                               
    if selected_optimisation_algorithm == 'optimised':
           #optimised parameters by NM
            
            #initial parameters                                           
            beta0=0.1610073221
            beta1=0.001423470338643
            beta2=  0.0737333545187117
            cha_time_beta= 21.732155377825364         
            gamma0=0.025909834357211
            gamma1=0.026700039069596
            gamma2=0.00008300
            gamma3=0.006066878694908
            cha_time_gamma= 12.359300607126448

            mu0=0.001510617364145 #0.07285434# #0.07278481
            mu1=0.000153164803984 #0.01222592# #0.01223569
            mu2=0.000080126119201#0.34390103# #0.34449327
            mu3= 0.000250643745785#0.02429897# #0.02447233
            cha_time_mu=26.359322685088163#5.31175157# #5.32643972

            sigma= 0.22303985 #0.30001983#0.30001935#0.15#0.5##0.49999927#0.21982829#
            delta= 0#0.02         
            p= 1.00000000#1#1.42728098##5 #1.42580770
            
            
            #define piecewise defined funtion for Beta_I and Beta_E
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
                    
             #Define Model 
            def ode_model(z, tspan,beta0, beta1, beta2, cha_time_beta,gamma0,gamma1,gamma2,gamma3,mu0,mu1,mu2,mu3,cha_time_mu, cha_time_gamma,sigma,p):

                S, E, I, R, D = z
                N = S + E + I + R + D
                dSdt = -(beta_I(tspan)*S*I)/N - (beta_E(tspan)*S*E)/N + delta*R
                dEdt = (beta_I(tspan)*S*I)/N + (beta_E(tspan)*S*E)/N - sigma*E
                dIdt = sigma*E - gamma(tspan)*I - mu(tspan)*I
                dRdt = gamma(tspan)*I - delta*R
                dDdt = mu(tspan)*I
                return [dSdt, dEdt, dIdt, dRdt, dDdt]
                
            params = Parameters()     
            params.add('beta0', value=beta0, min=0, max=1)
            params.add('beta1', value=beta1, min=0, max=1)
            params.add('beta2', value=beta2, min=0, max=1)
            params.add('cha_time_beta', value=cha_time_beta, min=0, max=100)
            params.add('gamma0', value=gamma0, min=0, max=1)
            params.add('gamma1', value=gamma1, min=0, max=1)
            params.add('gamma2', value=gamma2, min=0, max=1)
            params.add('gamma3', value=gamma3, min=0, max=1)
            params.add('mu0', value=mu0, min=0, max=1)
            params.add('mu1', value=mu1, min=0, max=1)
            params.add('mu2', value=mu2, min=0, max=1)
            params.add('mu3', value=mu3, min=0, max=1)
            params.add('cha_time_gamma', value=cha_time_gamma, min=0, max=100)
            params.add('cha_time_mu', value=cha_time_mu, min=0, max=100)
            params.add('sigma', value=sigma,min=0.07, max=0.5)# min=0.07, max=0.5)
            params.add('p', value=p,min=1, max=5)# min=0.07, max=0.5)
                 
            #define ODE Solver
            def ode_solver(tspan, initial_conditions, params):
                initS, initE, initI, initR, initD = initial_conditions
                beta0, beta1, beta2, cha_time_beta,gamma0,gamma1,gamma2,gamma3,mu0,mu1,mu2,mu3,cha_time_mu,\
                cha_time_gamma,sigma, p = params['beta0'].value, params['beta1'].value, \
                params['beta2'].value,  params['cha_time_beta'].value, params['gamma0'].value, params['gamma1'].value, \
                params['gamma2'].value,  params['gamma3'].value, params['mu0'].value, params['mu1'].value, \
                params['mu2'].value,  params['mu3'].value, params['cha_time_mu'].value,  params['cha_time_gamma'].value,\
                params['sigma'].value, params['p'].value
                initS = initN - (initE + initI + initR + initD)
                res = odeint(ode_model, [initS, initE, initI, initR, initD], tspan,args=(beta0, beta1, beta2, cha_time_beta, gamma0,gamma1,gamma2,
                                                                     gamma3,mu0,mu1,mu2,mu3,cha_time_gamma, cha_time_mu,sigma,p))
                return res

            initS = initN - (initE + initI + initR + initD)
            

            initial_conditions = [initS,initE, initI, initR, initD]
            #params['beta'].value, params['sigma'].value,params['gamma'].value, params['mu'].value = [beta, sigma, gamma, mu]
            tspan = np.arange(0, days+1, 1) #timespan,np.arrange to arrange day 0 till days with increment of 1
            sol = ode_solver(tspan, initial_conditions, params)
            S, E, I, R, D = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4]
       
            fig = go.Figure()
            #to able functionality of selectbox, put if statement
            fig.add_trace(go.Scatter(x=df2['date'][0:tRMCO], y=I[0:tRMCO], mode='lines',line_color='purple', name='Simulated'))
            fig.add_trace(go.Scatter(x=df2['date'][tRMCO:days+1], y=I[tRMCO:days+1], mode='lines',line_color='orange', name='Simulated'))
            fig.add_trace(go.Scatter(x=df2['date'], y=covid_history.iloc[0:days+1].daily_active_cases, mode='markers', marker_symbol='square',marker_color='purple',\
                                 name='Actual data', line = dict(dash='dash')))
        
         
            fig.update_layout(title='',
                xaxis_title='Days (year 2020)',
                yaxis_title='The number of active cases (I)',
                title_x=0.3, font_size= 22,
                width=700, height=500
                         )
            fig.update_xaxes(tickangle=-45, tickformat =  '%d<Br> %b <Br> ', tickmode='array',title_font=dict(size=18), tickfont=dict(size=16))
            fig.update_yaxes(title_font=dict(size=18), tickfont=dict(size=16))
            fig.update_layout(legend=dict(font=dict(size=18)))
            fig.update_layout(
            title='',
            margin=dict(t=0)) # Set the top margin to 0
            st.write(fig) 
            
            #st.markdown("##### Chosen time-windows: 27/2/2020-21/10/2020")
            text = "<i>Created by Norsyahidah Zulkarnain, Department of Computational & Theoretical Sciences, Kulliyyah of Science, IIUM</i>"         
            st.markdown(f"<span style='font-size:12px;'>{text}</span>", unsafe_allow_html=True)
    
   

st.set_page_config(page_title="GUI-SEIRD Predictive Model for COVID-19 📈", page_icon="📈", layout="centered")
st.markdown("## GUI-SEIRD Predictive Model for COVID-19 📈")
#st.sidebar.header("COVID-19 Predictor 📈")   
#st.write(
#    """COVID-19 predictor enables you to see how classical and modified SEIRD model with best-fit parameters fits the real cases of COVID-19 in Malaysia based on general cases of daily active infected (I), total recovered (R) and total death (D). 
#    """
#    ) 



COVID19_Predictor()
